#!/usr/bin/env python3
"""
Weighted Box Fusion (WBF) Ensemble Script with RoI Classifier Integration
=======================================================================

ClearSAR solution pipeline. Ensembles bounding-box predictions from multiple
object detection models using Weighted Boxes Fusion (WBF), then optionally
refines the fused scores with a trained RoI (ResNet-50 based) classifier.

Pipeline overview
-----------------
1. Load COCO-format prediction JSONs from N models (one list of dicts each).
2. Per image, convert every detection's [x, y, w, h] bbox to normalized
   [x1, y1, x2, y2] coordinates (using GT image dims when available).
3. Apply per-model temperature scaling to the raw confidence scores.
4. Fuse all models with `weighted_boxes_fusion` (weights + IoU threshold).
5. (Optional) Run a ResNet-50 RoI classifier over each fused box to refine
   scores using one of several strategies (multiply / replace / filter /
   insight_driven).
6. If a GT JSON is provided (val mode), compute COCO mAP before and after
   the classifier, and dump TP/FP statistics stratified by score range.

Usage
-----
    python ensemble_detections_wbf_roi_classifier.py --mode val
    python ensemble_detections_wbf_roi_classifier.py --mode test

The `--mode` flag selects the prediction set (val or test). Metrics are only
computed when a ground-truth JSON is available (val mode).
"""

import argparse
import json
import numpy as np
import os
import torch
import torch.nn as nn
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_align

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from ensemble_boxes import weighted_boxes_fusion
    HAS_WBF = True
except ImportError:
    HAS_WBF = False

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCOTOOLS = True
except ImportError:
    HAS_COCOTOOLS = False


class RoIClassifier(nn.Module):
    """ResNet-50 based region-of-interest classifier.

    Takes a batch of images and a list of RoI boxes per image, pools the
    ResNet-50 features at each box via RoIAlign, and predicts a scalar
    logit per box. A sigmoid over that logit yields the box's "valid
    detection" probability, used downstream to refine ensemble scores.
    """

    def __init__(self, output_size=(7, 7), spatial_scale=1/32.0):
        super().__init__()
        # ResNet-50 pretrained backbone, truncated before the final
        # classification layer (i.e. keep conv layers only -> 2048 channels).
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        # Small MLP head: pool RoI features -> 512 -> 1 logit.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, images, boxes_list):
        features = self.backbone(images)
        rois = roi_align(
            features, 
            boxes_list, 
            output_size=self.output_size, 
            spatial_scale=self.spatial_scale, 
            aligned=True
        )
        logits = self.head(rois)
        return logits.squeeze(-1)


def get_image_dims(gt_json_path):
    """Return {image_id: (width, height)} from a COCO GT JSON."""
    with open(gt_json_path, 'r') as f:
        data = json.load(f)
    return {img['id']: (img['width'], img['height']) for img in data['images']}

def load_gt_images(gt_json_path):
    """Return {image_id: image_info_dict} from a COCO GT JSON."""
    with open(gt_json_path, 'r') as f:
        data = json.load(f)
    return {img['id']: img for img in data['images']}


def apply_temperature_scaling(scores, temperature=1.0):
    """Apply temperature scaling to confidence scores.

    Converts probabilities to logits (log-odds), divides by the temperature,
    and maps back to probabilities via sigmoid. Temperature < 1 sharpens the
    distribution; temperature > 1 flattens it. This calibrates each model's
    confidence before WBF fusion.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    if isinstance(scores, list):
        scores = np.array(scores)
    scores = np.clip(scores, 1e-7, 1 - 1e-7)
    logits = np.log(scores / (1 - scores))
    scaled_logits = logits / temperature
    calibrated = 1.0 / (1.0 + np.exp(-scaled_logits))
    return calibrated.tolist() if isinstance(scores, np.ndarray) else calibrated


def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    if union_area < 1e-6:
        return 0.0
    return inter_area / union_area

def count_agreeing_models(fused_box, all_boxes_list, iou_threshold=0.5):
    agreeing_count = 0
    for model_boxes in all_boxes_list:
        for box in model_boxes:
            if compute_iou(fused_box, box) >= iou_threshold:
                agreeing_count += 1
                break  
    return agreeing_count

def evaluate_predictions(gt_path, predictions, title):
    """Compute COCO bbox mAP for `predictions` against `gt_path`.

    Returns mAP @ [0.5:0.95] (stats[0]), or None if GT is unavailable.
    """
    if not HAS_COCOTOOLS or not gt_path or not Path(gt_path).exists():
        print(f"\nSkipping COCO metrics for '{title}' (missing pycocotools or GT path).")
        return None
        
    print(f"\n{'='*50}\nEvaluating ({title})...\n{'='*50}")
    try:
        coco_gt = COCO(gt_path)
        img_ids_in_gt = set(coco_gt.getImgIds())
        
        filtered_dt_data = [d for d in predictions if d['image_id'] in img_ids_in_gt]
        if not filtered_dt_data:
            print("Error: No overlapping images between Predictions and GT.")
            return 0.0
            
        coco_dt = coco_gt.loadRes(filtered_dt_data)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0]
    except Exception as e:
        print(f"Error computing COCO metrics: {e}")
        return 0.0

def calculate_tp_fp_stats(filtered_preds, gt_path, iou_threshold=0.5):
    """
    Evaluates which predictions are Actual TPs vs Actual FPs (based on IoU with GT),
    then evaluates how well the classifier (based on refined scores) retains TPs and filters FPs.
    """
    if not gt_path or not Path(gt_path).exists():
        return
        
    with open(gt_path, 'r') as f:
        coco_data = json.load(f)
        
    gt_dict = defaultdict(list)
    for ann in coco_data['annotations']:
        x, y, w, h = ann['bbox']
        gt_dict[ann['image_id']].append([x, y, x + w, y + h])
        
    # Sort predictions by original score descending for proper evaluation 
    filtered_preds = sorted(filtered_preds, key=lambda x: x.get('orig_score', x['score']), reverse=True)
    
    preds_by_img = defaultdict(list)
    for p in filtered_preds:
        preds_by_img[p['image_id']].append(p)
        
    # We will stratify by original score ranges
    score_bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    
    tp_stats = {b: {'total': 0, 'kept': 0, 'dropped': 0} for b in score_bins}
    fp_stats = {b: {'total': 0, 'kept': 0, 'dropped': 0} for b in score_bins}
    
    total_actual_tp = 0
    total_actual_fp = 0
    
    for img_id, img_preds in preds_by_img.items():
        if img_id not in gt_dict:
            # Everything is FP if no GT for this image
            for p in img_preds:
                total_actual_fp += 1
                orig = p.get('orig_score', p['score'])
                clf_prob = p.get('clf_prob', 1.0 if p['score'] >= 0.5 else 0.0)
                for b in score_bins:
                    if b[0] <= orig < b[1]:
                        fp_stats[b]['total'] += 1
                        if clf_prob >= 0.5:
                            fp_stats[b]['kept'] += 1
                        else:
                            fp_stats[b]['dropped'] += 1
                        break
            continue
            
        gt_boxes = list(gt_dict[img_id])
        gt_matched = [False] * len(gt_boxes)
        
        for p in img_preds:
            px, py, pw, ph = p['bbox']
            pbox = [px, py, px+pw, py+ph]
            orig = p.get('orig_score', p['score'])
            clf_prob = p.get('clf_prob', 1.0 if p['score'] >= 0.5 else 0.0)
            
            best_iou = 0.0
            best_gt_idx = -1
            
            for g_idx, gbox in enumerate(gt_boxes):
                if gt_matched[g_idx]:
                    continue
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
                    
            is_tp = False
            if best_iou >= iou_threshold:
                gt_matched[best_gt_idx] = True
                is_tp = True
                total_actual_tp += 1
            else:
                total_actual_fp += 1
                
            stats_dict = tp_stats if is_tp else fp_stats
            for b in score_bins:
                if b[0] <= orig < b[1]:
                    stats_dict[b]['total'] += 1
                    if clf_prob >= 0.5:
                        stats_dict[b]['kept'] += 1
                    else:
                        stats_dict[b]['dropped'] += 1
                    break

    print("\n" + "="*85)
    print("CLASSIFIER IMPAACT STRATIFIED BY ORIGINAL DETECTION SCORE")
    print("Definition: 'Kept' = Classifier Prob >= 0.5 | 'Dropped' = Classifier Prob < 0.5")
    print("="*85)
    
    print("\nTRUE POSITIVES (Valid Detections) -> Goal: KEEp them (High Recall)")
    print(f"{'Orig Score Range':<18} | {'Total TPs':<12} | {'Kept (Good)':<14} | {'Dropped (Bad)':<14} | {'Recall':<10}")
    print("-" * 85)
    for b in score_bins:
        s = tp_stats[b]
        b_label = f"[{b[0]:.1f} - {1.0 if b[1]>1.0 else b[1]:.1f})"
        recall = (s['kept'] / s['total']) if s['total'] > 0 else 0.0
        print(f"{b_label:<18} | {s['total']:<12} | {s['kept']:<14} | {s['dropped']:<14} | {recall:.4f}")
        
    print("\nFALSE POSITIVES (Invalid Detections) -> Goal: DROP them (High Filter Rate)")
    print(f"{'Orig Score Range':<18} | {'Total FPs':<12} | {'Dropped (Good)':<14} | {'Kept (Bad)':<14} | {'Filter Rate':<10}")
    print("-" * 85)
    for b in score_bins:
        s = fp_stats[b]
        b_label = f"[{b[0]:.1f} - {1.0 if b[1]>1.0 else b[1]:.1f})"
        filter_rate = (s['dropped'] / s['total']) if s['total'] > 0 else 0.0
        print(f"{b_label:<18} | {s['total']:<12} | {s['dropped']:<14} | {s['kept']:<14} | {filter_rate:.4f}")
    print("="*85 + "\n")


def apply_roi_classifier(
        ensemble_results, 
        roi_checkpoint, 
        image_root, 
        gt_json_path,
        strategy="multiply", 
        device="cpu"):
    """Refine fused detections using the RoI classifier.

    For each image, crops the fused boxes, runs the classifier to get a
    per-box probability, and combines it with the original WBF score using
    the chosen `strategy`. Stores both `orig_score` and `clf_prob` on every
    prediction for downstream analysis.
    """
        
    print(f"\nInitializing RoI Classifier from {roi_checkpoint}...")
    model = RoIClassifier().to(device)
    model.load_state_dict(torch.load(roi_checkpoint, map_location=device))
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Map image_id -> {'file_name': ...} so we can locate raw images.
    images_info = {}
    if gt_json_path and Path(gt_json_path).exists():
        images_info = load_gt_images(gt_json_path)
    elif image_root and os.path.exists(image_root):
        for fname in os.listdir(image_root):
            stem = Path(fname).stem
            try:
                images_info[int(stem)] = {'file_name': fname}
            except ValueError:
                pass
            images_info[stem] = {'file_name': fname}

    preds_by_img = defaultdict(list)
    for p in ensemble_results:
        preds_by_img[p['image_id']].append(p)

    refined_preds = []
    
    print(f"Applying classifier to {len(preds_by_img)} images (Strategy: {strategy})...")
    with torch.no_grad():
        for img_id, img_preds in tqdm(preds_by_img.items()):
            if img_id not in images_info:
                refined_preds.extend(img_preds)
                continue
                
            file_name_info = images_info[img_id].get('file_name', '')
            if not file_name_info:
                refined_preds.extend(img_preds)
                continue
            
            img_path = os.path.join(image_root, file_name_info)
            if not os.path.exists(img_path):
                refined_preds.extend(img_preds)
                continue
                
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                refined_preds.extend(img_preds)
                continue
                
            image_t = transform(image).unsqueeze(0).to(device) 
            
            boxes = []
            for p in img_preds:
                x, y, w, h = p['bbox']
                boxes.append([x, y, x + w, y + h])
                
            boxes_t = torch.tensor(boxes, dtype=torch.float32).to(device)
            
            logits = model(image_t, [boxes_t])
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            probs = torch.sigmoid(logits)
            
            for i, p in enumerate(img_preds):
                new_p = p.copy()
                prob = probs[i].item()
                
                if strategy == "multiply":
                    new_p['score'] = float(p['score'] * prob)
                elif strategy == "replace":
                    new_p['score'] = float(prob)
                elif strategy == "filter":
                    new_p['score'] = float(p['score']) if prob >= 0.5 else 0.0001
                elif strategy == "insight_driven":
                    orig = p['score']
                    if orig < 0.3 and prob < 0.5:
                        # Low confidence + Classifier disagrees: Drop it
                        new_p['score'] = 0.2
                        
                new_p['orig_score'] = float(p['score'])
                new_p['clf_prob'] = float(prob)
                
                refined_preds.append(new_p)

    return refined_preds


def ensemble_wbf_roi(
    json_paths,
    output_path,
    gt_json_path=None,
    iou_thr=0.6,
    skip_box_thr=0.001,
    weights=None,
    temperature_values=None,
    image_root=None,
    roi_checkpoint=None,
    roi_strategy="multiply",
    roi_device="cuda:0"
):
    """Main ensemble entry point.

    Args:
        json_paths: list of paths to per-model COCO prediction JSONs.
        output_path: where the final fused predictions are written.
        gt_json_path: optional COCO GT JSON (enables image dims + metrics).
        iou_thr: IoU threshold used by WBF to match boxes across models.
        skip_box_thr: minimum fused confidence to keep a box.
        weights: per-model WBF fusion weights (default all 1.0).
        temperature_values: per-model temperature scaling (default all 1.0).
        image_root: directory of raw images for the RoI classifier.
        roi_checkpoint: path to the trained RoI classifier weights.
        roi_strategy: how classifier probabilities combine with WBF scores.
        roi_device: torch device for the classifier.
    """
    if not HAS_WBF:
        print("ERROR: Please install WBF first:  pip install ensemble-boxes")
        return
        
    # Use GT to recover exact image dimensions for bbox normalization.
    img_dims = {}
    if gt_json_path and Path(gt_json_path).exists():
        print("Loading GT to get exact image dimensions for normalization...")
        img_dims = get_image_dims(gt_json_path)

    num_models = len(json_paths)
    if weights is None:
        weights = [1] * num_models
    
    if temperature_values is None:
        temperature_values = [1.0] * num_models
        
    # image_preds[image_id][model_idx] -> list of detections.
    image_preds = defaultdict(lambda: [[] for _ in range(num_models)])
    
    print(f"Loading predictions from {num_models} models...")
    for model_idx, path in enumerate(json_paths):
        with open(path, 'r') as f:
            preds = json.load(f)
        for p in preds:
            if not p['bbox'] or p['score'] <= 0:
                continue
            image_preds[p['image_id']][model_idx].append(p)
            
    ensemble_results = []
    
    print(f"Applying WBF (IoU >= {iou_thr})...")
    for img_id, models_preds in tqdm(image_preds.items(), desc="Processing images"):
        if img_dims and img_id not in img_dims:
            continue
        W, H = img_dims.get(img_id, (100000.0, 100000.0))
        
        boxes_list, scores_list, labels_list = [], [], []
        
        # Build normalized box/scores/labels lists per model.
        for m_idx in range(num_models):
            m_boxes, m_scores, m_labels = [], [], []
            for p in models_preds[m_idx]:
                x, y, w, h = p['bbox']
                x1 = max(0.0, x / W)
                y1 = max(0.0, y / H)
                x2 = min(1.0, (x + w) / W)
                y2 = min(1.0, (y + h) / H)
                if x2 > x1 and y2 > y1:
                    m_boxes.append([x1, y1, x2, y2])
                    score = p['score']
                    # Apply per-model temperature scaling before fusion.
                    if temperature_values[m_idx] != 1.0:
                        score = apply_temperature_scaling([score], temperature_values[m_idx])[0]
                    m_scores.append(score)
                    m_labels.append(p['category_id'])
            
            boxes_list.append(m_boxes)
            scores_list.append(m_scores)
            labels_list.append(m_labels)
            
        # Fuse detections from all models into a single set per image.
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, 
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
        
        # Convert fused boxes back to absolute COCO [x, y, w, h] format.
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            score = min(score, 1.0) 
            x1, y1, x2, y2 = box
            
            ensemble_results.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x1*W), float(y1*H), float((x2-x1)*W), float((y2-y1)*H)],
                "score": float(score)
            })
            
    print(f"\nGenerated {len(ensemble_results)} WBF detections (Before Classifier).")
    
    # Metrics before classifier (only if GT is available).
    evaluate_predictions(gt_json_path, ensemble_results, "BEFORE CLASSIFIER")

    # Optionally refine with the RoI classifier and re-evaluate.
    if roi_checkpoint and os.path.exists(roi_checkpoint):
        refined_results = apply_roi_classifier(
            ensemble_results, 
            roi_checkpoint, 
            image_root, 
            gt_json_path, 
            strategy=roi_strategy,
            device=roi_device
        )
        
        evaluate_predictions(gt_json_path, refined_results, "AFTER CLASSIFIER")
        calculate_tp_fp_stats(refined_results, gt_json_path)
        
        with open(output_path, 'w') as f:
            json.dump(refined_results, f, indent=2)
        print(f"\nRefined predictions saved to {output_path}")
    else:
        print("\nNo RoI classifier applied. Saving raw ensemble output.")
        with open(output_path, 'w') as f:
            json.dump(ensemble_results, f, indent=2)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    # Per-model prediction JSONs (COCO detection format) for val and test.
    # Order matters: it must match the order of `weights` and
    # `temperature_values` below (one entry per model).
    #
    # Model order (index-aligned between val/test lists, weights and temps):
    #   0  Strip-R-CNN         1  DeIMv2           2  RF-DETR
    #   3  YOLO                4  DFine            5  CO-DETR
    #   6  DINO-DETR           7  RF-DETR (pseudo) 8  DDQ
    #   9  RTMDet             10  GLIP            11  DFine (exp08b)
    #  12  RF-DETR (pseudolabels, final)
    # ------------------------------------------------------------------
    val_json_paths = [
        "data/predictions/val/Strip-R-CNN.json",
        "data/predictions/val/DeIMv2.json",
        "data/predictions/val/RF_DETR.json",
        "data/predictions/val/YOLO.json",
        "data/predictions/val/DFine.json",
        "data/predictions/val/CO-DETR.json",
        "data/predictions/val/DINO-DETR.json",
        "data/predictions/val/RF_DETR_psudo.json",
        "data/predictions/val/DDQ.json",
        "data/predictions/val/RTMDet.json",
        "data/predictions/val/GLIP.json",
        "data/predictions/val/DFine_exp08b.json",
        "data/predictions/val/RF_DETR_pseudolabels.json",
    ]

    test_json_paths = [
        "data/predictions/test/Strip-R-CNN.json",
        "data/predictions/test/DeIMv2.json",
        "data/predictions/test/RF_DETR.json",
        "data/predictions/test/YOLO.json",
        "data/predictions/test/DFine.json",
        "data/predictions/test/CO-DETR.json",
        "data/predictions/test/DINO-DETR.json",
        "data/predictions/test/RF_DETR_psudo.json",
        "data/predictions/test/DDQ.json",
        "data/predictions/test/RTMDet.json",
        "data/predictions/test/GLIP.json",
        "data/predictions/test/DFine_exp08b.json",
        "data/predictions/test/RF_DETR_pseudolabels.json",
    ]

    parser = argparse.ArgumentParser(description="WBF ensemble with RoI classifier.")
    parser.add_argument("--mode", choices=["val", "test"], default="val",
                        help="Which prediction set to ensemble (default: val).")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (overrides the default per mode).")
    args = parser.parse_args()

    # Dataset paths (GT + raw images).
    gt_json_path = "data/gt/instances_val2017.json"
    val_images_root = "data/images/val2017"
    test_images_root = "data/images/test"

    # Trained RoI classifier checkpoint shipped with this repo.
    roi_ckpt = "checkpoints/roi_classifier_best.pth"

    # Per-model WBF weights and temperatures (index-aligned with json_paths).
    # The tuned values below were found via the finetune script.
    weights = [0.6, 0.0, 2.5, 3.8, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    temperature_values = [1.50, 0.90, 0.60, 1.00, 0.60, 1.30, 1.50, 0.90, 0.90, 0.60, 0.60, 0.60, 0.60]

    if args.mode == "val":
        json_paths = val_json_paths
        # Metrics are computed because a GT JSON is available.
        active_gt = gt_json_path
        active_image_root = val_images_root
        output_json_path = args.output or "data/ensemble_wbf_val_final.json"
    else:
        json_paths = test_json_paths
        # No GT for the test split -> metrics are skipped.
        active_gt = None
        active_image_root = test_images_root
        output_json_path = args.output or "data/ensemble_wbf_test_final.json"

    ensemble_wbf_roi(
        json_paths=json_paths,
        output_path=output_json_path,
        gt_json_path=active_gt,
        iou_thr=0.7,
        skip_box_thr=0.06,
        weights=weights,
        temperature_values=temperature_values,
        image_root=active_image_root,
        roi_checkpoint=roi_ckpt,
        roi_strategy="insight_driven",  # "multiply", "filter", "replace", or "insight_driven"
        roi_device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

