#!/usr/bin/env python3
"""
Greedy coordinate-wise finetuning of WBF ensemble weights and temperatures
=========================================================================

Finds per-model fusion weights and temperature scales that maximize the
validation mAP of the Weighted Boxes Fusion (WBF) ensemble.

The search runs in two **sequential** phases over every model, in order:

  1. **Weights**       : for each model, sweep `weight` over [0, 5]  step 0.1.
  2. **Temperatures**  : for each model, sweep `temperature` over
                         [0.1, 2.0] step 0.1 (temperature must be > 0).

Every time a candidate value increases the validation mAP (COCO bbox
mAP @ [0.5:0.95]), it is adopted. At the end the tuned weights and
temperatures are saved to `finetuned_params.json`.

Usage
-----
    python finetune_weights_temperatures.py \
        --predictions data/predictions/val/model1.json \
                       data/predictions/val/model2.json \
                       ... \
        --gt data/gt/instances_val2017.json

Notes
-----
- The number of `--predictions` paths defines the number of models.
- This is a greedy (not global) search: parameters are fixed one at a time.
- A full sweep is expensive: it re-runs WBF + COCO mAP for every candidate.
"""

import argparse
import json
import numpy as np
from collections import defaultdict

from tqdm import tqdm

from ensemble_boxes import weighted_boxes_fusion

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCOTOOLS = True
except ImportError:
    HAS_COCOTOOLS = False


# ----------------------------------------------------------------------
# Helpers (mirror the logic in ensemble_detections_wbf_roi_classifier.py)
# ----------------------------------------------------------------------
def get_image_dims(gt_json_path):
    """Return {image_id: (width, height)} from a COCO GT JSON."""
    with open(gt_json_path, 'r') as f:
        data = json.load(f)
    return {img['id']: (img['width'], img['height']) for img in data['images']}


def apply_temperature_scaling(scores, temperature=1.0):
    """Calibrate confidence scores with temperature scaling (log-odds / T)."""
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    if isinstance(scores, list):
        scores = np.array(scores)
    scores = np.clip(scores, 1e-7, 1 - 1e-7)
    logits = np.log(scores / (1 - scores))
    scaled_logits = logits / temperature
    calibrated = 1.0 / (1.0 + np.exp(-scaled_logits))
    return calibrated.tolist() if isinstance(scores, np.ndarray) else calibrated


def ensemble_wbf(json_paths, gt_path, weights, temperatures,
                 iou_thr=0.7, skip_box_thr=0.06):
    """Fuse predictions from all models with WBF.

    Returns a list of COCO-format detections
    ({"image_id", "category_id", "bbox": [x, y, w, h], "score"}).
    """
    img_dims = get_image_dims(gt_path)
    num_models = len(json_paths)

    # image_preds[image_id][model_idx] -> list of detections.
    image_preds = defaultdict(lambda: [[] for _ in range(num_models)])
    for model_idx, path in enumerate(json_paths):
        with open(path, 'r') as f:
            preds = json.load(f)
        for p in preds:
            if not p['bbox'] or p['score'] <= 0:
                continue
            image_preds[p['image_id']][model_idx].append(p)

    results = []
    for img_id, models_preds in image_preds.items():
        if img_id not in img_dims:
            continue
        W, H = img_dims[img_id]

        boxes_list, scores_list, labels_list = [], [], []

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
                    if temperatures[m_idx] != 1.0:
                        score = apply_temperature_scaling(
                            [score], temperatures[m_idx])[0]
                    m_scores.append(score)
                    m_labels.append(p['category_id'])

            boxes_list.append(m_boxes)
            scores_list.append(m_scores)
            labels_list.append(m_labels)

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )

        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1, x2, y2 = box
            results.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x1 * W), float(y1 * H),
                         float((x2 - x1) * W), float((y2 - y1) * H)],
                "score": float(min(score, 1.0)),
            })

    return results


def compute_map(gt_path, predictions):
    """Compute COCO bbox mAP @ [0.5:0.95] for `predictions` vs `gt_path`."""
    if not HAS_COCOTOOLS:
        raise RuntimeError("pycocotools is required. Install with: "
                           "pip install pycocotools")

    coco_gt = COCO(gt_path)
    img_ids = set(coco_gt.getImgIds())
    filtered = [d for d in predictions if d['image_id'] in img_ids]
    if not filtered:
        return 0.0

    coco_dt = coco_gt.loadRes(filtered)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]


def main():
    parser = argparse.ArgumentParser(
        description="Greedy finetuning of WBF weights and temperatures.")
    parser.add_argument('--predictions', nargs='+', required=True,
                        help='Paths to per-model COCO val prediction JSONs.')
    parser.add_argument('--gt', required=True,
                        help='Path to COCO GT JSON (val).')
    parser.add_argument('--iou-thr', type=float, default=0.7,
                        help='WBF IoU threshold (default 0.7).')
    parser.add_argument('--skip-box-thr', type=float, default=0.06,
                        help='WBF skip-box threshold (default 0.06).')
    parser.add_argument('--output', default='finetuned_params.json',
                        help='Where to save the tuned weights/temperatures.')
    args = parser.parse_args()

    num_models = len(args.predictions)

    # Start from a neutral configuration: all weights 1, all temps 1.
    weights = [1.0] * num_models
    temperatures = [1.0] * num_models

    def evaluate(w, t):
        preds = ensemble_wbf(args.predictions, args.gt, w, t,
                             args.iou_thr, args.skip_box_thr)
        return compute_map(args.gt, preds)

    print(f"Models: {num_models}")
    print("Computing baseline mAP (all weights=1.0, all temps=1.0)...")
    best_map = evaluate(weights, temperatures)
    print(f"Baseline mAP: {best_map:.4f}")

    # ------------------------------------------------------------------
    # Phase 1: weights — sweep [0, 5] step 0.1 for each model, in order.
    # ------------------------------------------------------------------
    weight_grid = np.arange(0.0, 5.0 + 1e-9, 0.1)
    print("\n=== Phase 1: tuning model weights ===")
    for i in range(num_models):
        best_w = weights[i]
        best_m = best_map
        for w in tqdm(weight_grid, desc=f"Weight model {i}", leave=False):
            trial_weights = weights.copy()
            trial_weights[i] = float(w)
            m = evaluate(trial_weights, temperatures)
            if m > best_m:
                best_m = m
                best_w = float(w)
                print(f"  [weights] model {i}: weight={best_w:.1f} "
                      f"-> mAP {best_m:.4f} (improved)")
        weights[i] = best_w
        if best_m > best_map:
            best_map = best_m
        print(f"[weights] model {i} final: {best_w:.1f} | best mAP {best_map:.4f}")

    # ------------------------------------------------------------------
    # Phase 2: temperatures — sweep [0.1, 2.0] step 0.1 per model.
    # ------------------------------------------------------------------
    temp_grid = np.arange(0.1, 2.0 + 1e-9, 0.1)
    print("\n=== Phase 2: tuning model temperatures ===")
    for i in range(num_models):
        best_t = temperatures[i]
        best_m = best_map
        for t in tqdm(temp_grid, desc=f"Temp model {i}", leave=False):
            trial_temps = temperatures.copy()
            trial_temps[i] = float(t)
            m = evaluate(weights, trial_temps)
            if m > best_m:
                best_m = m
                best_t = float(t)
                print(f"  [temps] model {i}: temp={best_t:.1f} "
                      f"-> mAP {best_m:.4f} (improved)")
        temperatures[i] = best_t
        if best_m > best_map:
            best_map = best_m
        print(f"[temps] model {i} final: {best_t:.1f} | best mAP {best_map:.4f}")

    # ------------------------------------------------------------------
    # Save results.
    # ------------------------------------------------------------------
    result = {
        "num_models": num_models,
        "weights": [round(w, 1) for w in weights],
        "temperature_values": [round(t, 1) for t in temperatures],
        "best_map": round(best_map, 6),
    }
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print("\n=== Finetuning complete ===")
    print(f"Best mAP: {best_map:.4f}")
    print(f"Weights:       {[round(w, 1) for w in weights]}")
    print(f"Temperatures:  {[round(t, 1) for t in temperatures]}")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
