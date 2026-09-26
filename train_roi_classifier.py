#!/usr/bin/env python3
"""
Train the RoI (ResNet-50) classifier that refines the fused ensemble scores
===========================================================================

Reproduces the checkpoint `checkpoints/roi_classifier_best.pth` consumed by
`ensemble_detections_wbf_roi_classifier.py`, following the team's description
(see `docs/model_training_specs.md`, section "RoI classifier"):

  1. Build candidate boxes by running the ensemble pipeline (temperature
     scaling + Weighted Boxes Fusion) on the images you care about, or load a
     precomputed list of fused candidate detections (`--candidates`).
  2. Label every candidate: a candidate whose max IoU with a ground-truth box
     is >= 0.5 is a true positive (label 1), every other candidate a false
     positive (label 0).
  3. Downsample the majority class so the two classes are balanced.
  4. Split the candidates by image into a training split and a held-out
     validation split.
  5. Fine-tune a torchvision ResNet-50 (`IMAGENET1K_V2` weights) truncated
     before avgpool/fc, with RoIAlign (7x7, spatial scale 1/32, aligned) and
     the MLP head used by the ensemble script. The whole network is trained.
  6. Save the best checkpoint (raw `model.state_dict()`) exactly in the format
     the ensemble script loads, and write a sidecar summary JSON.

Training configuration:

  - loss          : BCEWithLogitsLoss on the per-box logits
  - optimizer     : Adam, lr 1e-4, weight decay 1e-4
  - schedule      : cosine annealing of the learning rate over all epochs
  - epochs        : 50 (best checkpoint on the validation split is kept)
  - batch         : all boxes of one image per forward/backward pass, with
                    gradient accumulation across `--batch-images` images
  - augmentation  : none (boxes are aligned with the raw image)
  - normalization : pixels to [0,1] then ImageNet mean/std, image not resized
  - validation    : ~10 % of the images that contain candidates, seed 42
  - label IoU     : 0.5 (a box overlapping a GT box at IoU >= 0.5 is a TP)

Usage
-----
Train from the per-model prediction files of a split (fuses them first):

    python train_roi_classifier.py \
        --images /path/to/images/train \
        --annotations data/gt/instances_train.json \
        --predictions data/predictions/train/Strip-R-CNN.json \
                      data/predictions/train/DeIMv2.json \
                      ... (all 13) \
        --weights 0.6,0.0,2.5,3.8,3.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 \
        --temperatures 1.5,0.9,0.6,1.0,0.6,1.3,1.5,0.9,0.9,0.6,0.6,0.6,0.6 \
        --output checkpoints/roi_classifier_best.pth

Or train from an already-fused list of candidate detections (COCO format):

    python train_roi_classifier.py \
        --images /path/to/images/train \
        --annotations data/gt/instances_train.json \
        --candidates candidates.json \
        --output checkpoints/roi_classifier_best.pth
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_align

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ---------------------------------------------------------------------------
# Architecture — must stay in sync with `RoIClassifier` in
# `ensemble_detections_wbf_roi_classifier.py`. The saved checkpoint is loaded
# there with `model.load_state_dict(...)`, so the module names must match.
# ---------------------------------------------------------------------------
class RoIClassifier(nn.Module):
    """ResNet-50 based region-of-interest classifier.

    Takes a batch of images and a tensor/list of RoI boxes per image, pools
    the ResNet-50 features at each box via RoIAlign, and predicts a scalar
    logit per box. A sigmoid over that logit yields the box's "valid
    detection" probability.
    """

    def __init__(self, output_size=(7, 7), spatial_scale=1 / 32.0):
        super().__init__()
        # ResNet-50 pretrained backbone, truncated before the final
        # classification layer (keep conv layers only -> 2048 channels).
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
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
            nn.Linear(512, 1),
        )

    def forward(self, images, boxes_list):
        features = self.backbone(images)
        rois = roi_align(
            features,
            boxes_list,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            aligned=True,
        )
        logits = self.head(rois)
        return logits.squeeze(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_iou(box1, box2):
    """IoU of two [x1, y1, x2, y2] boxes."""
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


def apply_temperature_scaling(scores, temperature=1.0):
    """Calibrate confidence scores with temperature scaling (logits / T)."""
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    if isinstance(scores, list):
        scores = np.array(scores)
    scores = np.clip(scores, 1e-7, 1 - 1e-7)
    logits = np.log(scores / (1 - scores))
    scaled_logits = logits / temperature
    calibrated = 1.0 / (1.0 + np.exp(-scaled_logits))
    return calibrated.tolist() if isinstance(scores, np.ndarray) else calibrated


def get_image_dims(gt_json_path):
    """Return {image_id: (width, height)} from a COCO GT JSON."""
    with open(gt_json_path, "r") as f:
        data = json.load(f)
    return {img["id"]: (img["width"], img["height"]) for img in data["images"]}


def fusion_candidates(prediction_paths, gt_path, weights, temperatures,
                      iou_thr=0.7, skip_box_thr=0.06):
    """Fuse per-model prediction files into one list of candidate detections.

    Mirrors the fusion loop in `ensemble_detections_wbf_roi_classifier.py`:
    drop empty/zero-score boxes, normalize with GT image dims, apply per-model
    temperature scaling, run WBF, convert the fused boxes back to absolute
    COCO [x, y, w, h] detections.
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        raise RuntimeError(
            "ensemble-boxes is required to fuse --predictions. Install it "
            "with: pip install ensemble-boxes"
        )

    img_dims = get_image_dims(gt_path)
    num_models = len(prediction_paths)
    weights = [1.0] * num_models if weights is None else weights
    temperatures = [1.0] * num_models if temperatures is None else temperatures

    image_preds = defaultdict(lambda: [[] for _ in range(num_models)])
    for model_idx, path in enumerate(prediction_paths):
        with open(path, "r") as f:
            preds = json.load(f)
        for p in preds:
            if not p["bbox"] or p["score"] <= 0:
                continue
            image_preds[p["image_id"]][model_idx].append(p)

    candidates = []
    for img_id, models_preds in tqdm(image_preds.items(), desc="Fusing"):
        if img_id not in img_dims:
            continue
        W, H = img_dims[img_id]

        boxes_list, scores_list, labels_list = [], [], []
        for m_idx in range(num_models):
            m_boxes, m_scores, m_labels = [], [], []
            for p in models_preds[m_idx]:
                x, y, w, h = p["bbox"]
                x1 = max(0.0, x / W)
                y1 = max(0.0, y / H)
                x2 = min(1.0, (x + w) / W)
                y2 = min(1.0, (y + h) / H)
                if x2 > x1 and y2 > y1:
                    m_boxes.append([x1, y1, x2, y2])
                    score = p["score"]
                    if temperatures[m_idx] != 1.0:
                        score = apply_temperature_scaling(
                            [score], temperatures[m_idx])[0]
                    m_scores.append(score)
                    m_labels.append(p["category_id"])
            boxes_list.append(m_boxes)
            scores_list.append(m_scores)
            labels_list.append(m_labels)

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr,
        )
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1, x2, y2 = box
            candidates.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x1 * W), float(y1 * H),
                         float((x2 - x1) * W), float((y2 - y1) * H)],
                "score": float(min(score, 1.0)),
            })
    return candidates


def load_annotations(annotations_path):
    """Return (image_id -> (w, h), image_id -> file_name, image_id -> gt boxes)."""
    with open(annotations_path, "r") as f:
        data = json.load(f)

    img_sizes = {}
    img_files = {}
    for img in data["images"]:
        img_sizes[img["id"]] = (img["width"], img["height"])
        img_files[img["id"]] = img.get("file_name", f"{img['id']}.png")

    gt_by_img = defaultdict(list)
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        gt_by_img[ann["image_id"]].append([x, y, x + w, y + h])
    return img_sizes, img_files, gt_by_img


def label_candidates(candidates, gt_by_img, label_threshold=0.5):
    """Return a list of (candidate, label) pairs."""
    labeled = []
    for c in candidates:
        x, y, w, h = c["bbox"]
        box = [x, y, x + w, y + h]
        gts = gt_by_img.get(c["image_id"], [])
        best = max((compute_iou(box, g) for g in gts), default=0.0)
        labeled.append((c, 1.0 if best >= label_threshold else 0.0))
    return labeled


def balance_classes(labeled, seed):
    """Downsample the larger class so the two classes are balanced."""
    tp = [x for x in labeled if x[1] == 1.0]
    fp = [x for x in labeled if x[1] == 0.0]
    keep = min(len(tp), len(fp))
    rng = random.Random(seed)
    if len(fp) > keep:
        fp = rng.sample(fp, keep)
    if len(tp) > keep:
        tp = rng.sample(tp, keep)
    return tp + fp


def main():
    parser = argparse.ArgumentParser(
        description="Train the RoI (ResNet-50) classifier for the ClearSAR "
                    "WBF ensemble.")
    parser.add_argument("--images", required=True,
                        help="Directory with the raw images referenced by "
                             "--annotations.")
    parser.add_argument("--annotations", required=True,
                        help="COCO ground-truth JSON (used for labels, image "
                             "dims and file names).")
    parser.add_argument("--candidates", default=None,
                        help="Precomputed fused detection list (COCO format) "
                             "to train on. Mutually exclusive with "
                             "--predictions.")
    parser.add_argument("--predictions", nargs="+", default=None,
                        help="Per-model COCO prediction JSONs of the split; "
                             "fused with WBF to build candidates. Mutually "
                             "exclusive with --candidates.")
    parser.add_argument("--weights", default=None,
                        help="Comma-separated WBF weights used when fusing "
                             "--predictions (default: all 1.0).")
    parser.add_argument("--temperatures", default=None,
                        help="Comma-separated temperature values used when "
                             "fusing --predictions (default: all 1.0).")
    parser.add_argument("--iou-thr", type=float, default=0.7,
                        help="WBF IoU threshold used when fusing (default 0.7).")
    parser.add_argument("--skip-box-thr", type=float, default=0.06,
                        help="WBF minimum scaled score (default 0.06).")
    parser.add_argument("--label-threshold", type=float, default=0.5,
                        help="IoU with a GT box that makes a candidate a TP "
                             "(default 0.5).")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of candidate images held out for "
                             "validation (default 0.1).")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default 50).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate, Adam (default 1e-4).")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Adam weight decay (default 1e-4).")
    parser.add_argument("--batch-images", type=int, default=1,
                        help="Images per optimizer step via gradient "
                             "accumulation (default 1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42).")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available).")
    parser.add_argument("--output", default="checkpoints/roi_classifier_best.pth",
                        help="Where to write the best checkpoint "
                             "(default checkpoints/roi_classifier_best.pth).")
    args = parser.parse_args()

    if (args.candidates is None) == (args.predictions is None):
        parser.error("Exactly one of --candidates or --predictions is required.")

    weights = None
    temperatures = None
    if args.weights is not None:
        weights = [float(x) for x in args.weights.split(",") if x.strip()]
    if args.temperatures is not None:
        temperatures = [float(x) for x in args.temperatures.split(",") if x.strip()]

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Device: {device}")

    # --- Load annotations ---------------------------------------------------
    img_sizes, img_files, gt_by_img = load_annotations(args.annotations)
    print(f"Ground truth: {len(img_sizes)} images, "
          f"{sum(len(v) for v in gt_by_img.values())} boxes")

    # --- Candidate boxes ----------------------------------------------------
    if args.candidates is not None:
        print(f"Loading candidate detections from {args.candidates} ...")
        with open(args.candidates, "r") as f:
            candidates = json.load(f)
    else:
        if weights is not None and len(weights) != len(args.predictions):
            parser.error("--weights must have one value per --predictions path.")
        if temperatures is not None and len(temperatures) != len(args.predictions):
            parser.error("--temperatures must have one value per --predictions path.")
        print(f"Fusing {len(args.predictions)} prediction files ...")
        candidates = fusion_candidates(
            args.predictions, args.annotations, weights, temperatures,
            args.iou_thr, args.skip_box_thr)
    print(f"Candidate boxes: {len(candidates)}")

    # --- Labeling -----------------------------------------------------------
    labeled = label_candidates(candidates, gt_by_img, args.label_threshold)
    n_tp = sum(1 for _, l in labeled if l == 1.0)
    n_fp = sum(1 for _, l in labeled if l == 0.0)
    print(f"Labeled: {n_tp} true positives, {n_fp} false positives")

    if n_tp == 0:
        raise SystemExit("No true positives found — check --annotations and "
                         "--label-threshold.")
    if n_fp == 0:
        raise SystemExit("No false positives found — cannot balance classes.")

    balanced = balance_classes(labeled, args.seed)
    print(f"After downsampling the majority class: "
          f"{sum(1 for _, l in balanced if l == 1.0)} TP + "
          f"{sum(1 for _, l in balanced if l == 0.0)} FP = {len(balanced)}")

    # --- Split by image -----------------------------------------------------
    boxes_by_img = defaultdict(list)
    for c, label in balanced:
        boxes_by_img[c["image_id"]].append((c, label))
    all_imgs = sorted(boxes_by_img.keys())
    random.Random(args.seed).shuffle(all_imgs)
    n_val = max(1, int(round(len(all_imgs) * args.val_fraction)))
    val_imgs = set(all_imgs[:n_val])
    train_imgs = set(all_imgs[n_val:])

    def split_stats(imgs):
        return sum(len(boxes_by_img[i]) for i in imgs)

    print(f"Split: {len(train_imgs)} train images "
          f"({split_stats(train_imgs)} boxes), "
          f"{len(val_imgs)} validation images "
          f"({split_stats(val_imgs)} boxes)")

    # --- Image transform (must match the ensemble script) --------------------
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def load_image(img_id):
        path = os.path.join(args.images, img_files.get(img_id, f"{img_id}.png"))
        image = Image.open(path).convert("RGB")
        return transform(image).unsqueeze(0).to(device)

    def make_tensors(img_id):
        """(boxes [N,4] xyxy pixels, labels [N], scores [N])."""
        W, H = img_sizes.get(img_id, (0, 0))
        boxes, labels, scores = [], [], []
        for c, label in boxes_by_img[img_id]:
            x, y, w, h = c["bbox"]
            if W and H:
                x1 = min(max(x, 0.0), W)
                y1 = min(max(y, 0.0), H)
                x2 = min(max(x + w, 0.0), W)
                y2 = min(max(y + h, 0.0), H)
            else:
                x1, y1, x2, y2 = x, y, x + w, y + h
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(label)
                scores.append(c.get("score", 0.0))
        bt = torch.tensor(boxes, dtype=torch.float32).to(device)
        lt = torch.tensor(labels, dtype=torch.float32).to(device)
        st = torch.tensor(scores, dtype=torch.float32).to(device)
        return bt, lt, st

    # --- Model / optimizer / loss --------------------------------------------
    model = RoIClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    def evaluate(imgs):
        model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        with torch.no_grad():
            for img_id in imgs:
                bt, lt, _ = make_tensors(img_id)
                if bt.shape[0] == 0:
                    continue
                image_t = load_image(img_id)
                logits = model(image_t, [bt])
                total_loss += criterion(logits, lt).item() * len(lt)
                probs = torch.sigmoid(logits)
                total_correct += ((probs >= 0.5).float() == lt).sum().item()
                total += len(lt)
        model.train()
        if total == 0:
            return 0.0, 0.0
        return total_loss / total, total_correct / total

    # --- Training loop --------------------------------------------------------
    best_val_acc = -1.0
    best_epoch = -1
    output_path = Path(args.output)

    print(f"\nTraining for {args.epochs} epochs "
          f"(Adam lr={args.lr}, wd={args.weight_decay}, "
          f"{args.batch_images} image(s) per optimizer step) ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
        optimizer.zero_grad()
        train_list = list(train_imgs)
        random.Random(args.seed + epoch).shuffle(train_list)

        for step, img_id in enumerate(train_list):
            bt, lt, _ = make_tensors(img_id)
            if bt.shape[0] == 0:
                continue
            image_t = load_image(img_id)
            logits = model(image_t, [bt])
            loss = criterion(logits, lt)
            loss.backward()

            epoch_loss += loss.item() * len(lt)
            probs = torch.sigmoid(logits.detach())
            epoch_correct += ((probs >= 0.5).float() == lt).sum().item()
            epoch_total += len(lt)

            if (step + 1) % args.batch_images == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Flush any remaining accumulated gradients.
        if (len(train_list)) % args.batch_images != 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        val_loss, val_acc = evaluate(val_imgs)
        lr_now = optimizer.param_groups[0]["lr"]

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)

        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"lr {lr_now:.2e} | best val acc {best_val_acc:.4f} "
              f"(epoch {best_epoch}){' *' if improved else ''}")

    # --- Save summary --------------------------------------------------------
    summary = {
        "candidates_total": len(candidates),
        "true_positives": n_tp,
        "false_positives": n_fp,
        "balanced_boxes": len(balanced),
        "train_images": len(train_imgs),
        "train_boxes": split_stats(train_imgs),
        "val_images": len(val_imgs),
        "val_boxes": split_stats(val_imgs),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "architecture": {"backbone": "resnet50_IMAGENET1K_V2",
                         "roi_align": {"output_size": [7, 7],
                                       "spatial_scale": 1 / 32.0,
                                       "aligned": True},
                         "head": "AdaptiveAvgPool->Linear(2048,512)->ReLU"
                                 "->Dropout(0.5)->Linear(512,1)"},
        "hyperparameters": {"epochs": args.epochs, "lr": args.lr,
                            "weight_decay": args.weight_decay,
                            "batch_images": args.batch_images,
                            "seed": args.seed,
                            "val_fraction": args.val_fraction,
                            "label_threshold": args.label_threshold},
    }
    summary_path = output_path.with_name(output_path.stem + ".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch "
          f"{best_epoch}.")
    print(f"Checkpoint saved to {output_path}")
    print(f"Summary written to {summary_path}")
    print("Note: check that the checkpoint loads with "
          "ensemble_detections_wbf_roi_classifier.py (it expects a plain "
          "state dict) by running the ensemble script on the validation split.")


if __name__ == "__main__":
    main()