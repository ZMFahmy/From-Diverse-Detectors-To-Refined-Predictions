# ClearSAR — WBF Ensemble with RoI Classifier

> **Winner of the [ClearSAR challenge](https://platform.ai4eo.eu/clear-sar).**
> This solution achieved **first place on both the public and private test sets**
> of the ClearSAR Track 1 leaderboard.

- **Paper:** _From Diverse Detectors to Refined Predictions: A Multi-Evidence
  Pipeline for Sentinel-1 SAR RFI Detection_ — ClearSAR Track 1 Challenge,
  IEEE ICIP 2026.

This repository contains the ClearSAR solution pipeline: an ensemble of object
detection models fused with **Weighted Boxes Fusion (WBF)**, followed by an
optional **RoI (ResNet-50) classifier** that refines the fused scores. The
shipped predictions are produced by 13 detectors spanning **10 architectures**
(CNN- and transformer-based).

The pipeline consumes per-model predictions in **COCO detection format** and
produces a single fused prediction file (also COCO format). For the validation
split it can additionally compute COCO metrics (mAP) and TP/FP statistics.

---

## Repository layout

```
clearsar_ensemble_solution/
├── ensemble_detections_wbf_roi_classifier.py   # Main ensemble + classifier script
├── finetune_weights_temperatures.py            # Greedy tuning of weights & temperatures
├── train_roi_classifier.py                     # Trains the RoI classifier checkpoint
├── README.md
├── docs/
│   └── model_training_specs.md                 # Training specs of the 13 detectors + classifier
├── checkpoints/
│   └── roi_classifier_best.pth                 # Trained RoI classifier (ResNet-50 based)
└── data/
    ├── gt/
    │   └── instances_val2017.json              # Validation ground truth (COCO)
    ├── predictions/
    │   ├── val/                                # Per-model validation predictions (13 models)
    │   │   ├── Strip-R-CNN.json
    │   │   ├── DeIMv2.json
    │   │   ├── RF_DETR.json
    │   │   ├── YOLO.json
    │   │   ├── DFine.json
    │   │   ├── CO-DETR.json
    │   │   ├── DINO-DETR.json
    │   │   ├── RF_DETR_psudo.json
    │   │   ├── DDQ.json
    │   │   ├── RTMDet.json
    │   │   ├── GLIP.json
    │   │   ├── DFine_exp08b.json
    │   │   └── RF_DETR_pseudolabels.json
    │   └── test/                               # Per-model test predictions (same 13 models)
    │       └── ... (same names as val)
    ├── ensemble_wbf_val_final.json             # Final fused val predictions (example output)
    └── ensemble_wbf_test_final.json            # Final fused test predictions (example output)
```

> **Note on images:** the RoI classifier needs the raw images to crop each
> fused box. The image directories are **not** committed to the repo (they are
> large). Place them at `data/images/val2017` and `data/images/test`, or edit
> the `val_images_root` / `test_images_root` constants in the script's
> `if __name__ == "__main__"` block.

---

## Checkpoints and data

The trained artifacts (`checkpoints/`) and the dataset folder (`data/`) are
**not** stored in the repository; the folders stay in the repo with placeholder
files so the layout is preserved. Download the actual content from Google
Drive and extract each folder over its matching repository directory:

| Repo directory | Contents                                                                                                                                                                                                                     | Download                                                                                                              |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `checkpoints/` | Trained RoI classifier (`roi_classifier_best.pth`)                                                                                                                                                                           | [checkpoints — Google Drive](https://drive.google.com/drive/folders/15p7AfL-z3JUKztCh9RzzjVqk3b4QyezM?usp=drive_link) |
| `data/`        | COCO ground truth (`data/gt/`), per-model predictions (`data/predictions/val/` and `data/predictions/test/`, 13 files each), example fused outputs (`data/ensemble_wbf_val_final.json`, `data/ensemble_wbf_test_final.json`) | [data — Google Drive](https://drive.google.com/drive/folders/1Du6ylVW-wggJi3yFgZtzk7Kl05YxAZ9_?usp=drive_link)        |

The scripts read these exact paths, so extract the downloads **in place** — do
not rename or move files:

- `checkpoints/roi_classifier_best.pth`
- `data/gt/instances_val2017.json`
- `data/predictions/val/*.json` (13 files) and `data/predictions/test/*.json`
  (13 files)
- `data/ensemble_wbf_val_final.json` and `data/ensemble_wbf_test_final.json`

---

## Prediction JSON format (COCO detection format)

Every model must output its predictions as a JSON **list of detection dicts**.
Each detection has the following fields:

```json
[
  {
    "image_id": 1006,
    "category_id": 1,
    "bbox": [15.61, 138.95, 21.96, 9.23],
    "score": 0.770805
  }
]
```

| Field         | Type    | Description                                    |
| ------------- | ------- | ---------------------------------------------- |
| `image_id`    | int     | COCO image id the box belongs to               |
| `category_id` | int     | Class id of the detection                      |
| `bbox`        | [float] | `[x, y, width, height]` in **absolute pixels** |
| `score`       | float   | Detection confidence in `[0, 1]`               |

The ground-truth file must be a standard COCO annotation JSON with the usual
`images`, `annotations`, and `categories` sections:

```json
{
  "images": [
    { "id": 1132, "width": 517, "height": 341, "file_name": "1132.png" }
  ],
  "annotations": [
    {
      "id": 28,
      "category_id": 1,
      "image_id": 1132,
      "bbox": [72, 284, 103, 16],
      "area": 1648,
      "iscrowd": 0
    }
  ],
  "categories": [{ "id": 1, "name": "RFI" }]
}
```

---

## Pipeline flow

```
per-model COCO predictions (N JSONs)
        │
        ▼
┌─────────────────────────────────────────────┐
│ 1. Normalize boxes                          │  bbox [x,y,w,h] -> [x1,y1,x2,y2]/[W,H]
│    (uses GT image dims when available)      │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ 2. Temperature scaling (per model)          │  logit = log(p/(1-p)); logit/T
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ 3. Weighted Boxes Fusion (WBF)              │  weighted_boxes_fusion(weights, iou_thr)
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ 4. RoI classifier refinement (optional)     │  ResNet-50 RoIAlign -> sigmoid prob
│    strategies: multiply / replace / filter /│  combine prob with fused score
│                insight_driven               │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ 5. Save fused predictions (COCO format)     │
│    + (val only) compute COCO mAP & TP/FP    │
└─────────────────────────────────────────────┘
```

### Step-by-step

1. **Load & filter.** Each model's predictions are read and grouped by
   `image_id`. Rows with an empty box or a score of `0` or less are dropped.

2. **Normalize boxes.** Every `[x, y, w, h]` box is converted to normalized
   `[x1, y1, x2, y2]` coordinates in `[0, 1]` and clipped:

   - On **val**, the exact image dimensions come from the ground-truth JSON.
   - On **test** no annotation file is used, so coordinates are divided by a
     fixed value of `100_000` and multiplied back after fusion (clipping then
     only ever acts at 0).
   - Boxes with zero width or zero height are dropped.

3. **Temperature scaling.** Each model's raw confidence is recalibrated with a
   per-model temperature `T`:
   `p' = sigmoid(log(p/(1-p)) / T)`, with scores first clipped to
   `1e-7 .. 1-1e-7` so the logit stays finite. `T < 1` sharpens, `T > 1`
   flattens. `T = 1` is a no-op.

4. **WBF fusion.** All models are fused per image with
   `weighted_boxes_fusion` from the `ensemble-boxes` package. The submitted run
   uses `iou_thr = 0.7`, `skip_box_thr = 0.06` (applied _after_ temperature
   scaling) and `conf_type = 'avg'`. The fused score rewards agreement between
   sources (paper eq. 1):

   `s* = ( Σ_{i∈C} w_{m(i)} s_i / Σ_{i∈C} w_{m(i)} ) · P/N`

   where the sum runs over the `P = |C|` boxes of a fused cluster, `s_i` is a
   temperature-scaled score and `N` is the total number of models; the factor
   `P/N` penalises low-agreement clusters. Sources with `w_m = 0` do not affect
   the geometry or the confidence of a fused box, but their boxes still join a
   cluster and increment `P` — contributing to the agreement signal. Scores
   above 1 are clamped to 1 after fusion.

5. **RoI classifier (optional).** A ResNet-50 backbone crops each fused box via
   RoIAlign and predicts a scalar "valid detection" (not a FP) probability. That
   probability is combined with the fused score using the chosen strategy:

   | Strategy         | Effect                                                                                                 |
   | ---------------- | ------------------------------------------------------------------------------------------------------ |
   | `multiply`       | `score = fused_score * clf_prob`                                                                       |
   | `replace`        | `score = clf_prob`                                                                                     |
   | `filter`         | `score = fused_score` if `clf_prob >= 0.5` else `1e-4`                                                 |
   | `insight_driven` | `score = 0.2` if `fused_score < 0.3` and `clf_prob < 0.5`, else unchanged — this is the submitted rule |

   The classifier never moves or removes boxes; only scores change. Every
   output box keeps its pre-classifier score as `orig_score` and its classifier
   probability as `clf_prob`.

6. **Output & metrics.** The fused predictions are written as a COCO detection
   list. In **val** mode, COCO bbox mAP is reported _before_ and _after_ the
   classifier, and a TP/FP breakdown stratified by the original detection score
   is printed.

### Fusion specifications (submitted run)

| Parameter           | Value                                                                             |
| ------------------- | --------------------------------------------------------------------------------- |
| Sources             | 13 prediction files (see [Models used](#models-used-for-the-shipped-predictions)) |
| Weights             | `[0.6, 0.0, 2.5, 3.8, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`               |
| Temperatures        | `[1.5, 0.9, 0.6, 1.0, 0.6, 1.3, 1.5, 0.9, 0.9, 0.6, 0.6, 0.6, 0.6]`               |
| Fused-score formula | `s* = (Σ w·s / Σ w) · P/N` with `P/N` agreement penalty (paper eq. 1)             |
| `iou_thr`           | `0.7`                                                                             |
| `skip_box_thr`      | `0.06` (applied after temperature scaling)                                        |
| `conf_type`         | `avg` (ensemble-boxes default)                                                    |
| Rescoring rule      | `score = 0.2` if `fused_score < 0.3` and `clf_prob < 0.5`, else unchanged         |

> A weight of `0.0` does not influence the geometry or the confidence of a
> fused box, but the source's boxes still join the fusion and increment the
> agreement count `P`, as described in the paper.

### The RoI classifier

- **Backbone:** torchvision ResNet-50 up to and including `layer4` (no avgpool,
  no fc), 2048 channels at stride 32; about 24.6M parameters in total.
- **Input:** the full image at original size, Pillow, RGB, pixels scaled to
  `[0, 1]`, then ImageNet mean/std normalization. The image is not resized or
  padded; one image is processed at a time with all of its boxes.
- **RoIAlign:** `torchvision.ops.roi_align`, output size `7x7`, spatial scale
  `1/32`, `aligned=True`.
- **Head:** adaptive average pooling to a 2048-d vector, `Linear(2048, 512)`,
  ReLU, `Dropout(0.5)`, `Linear(512, 1)`, sigmoid.
- **Checkpoint:** `checkpoints/roi_classifier_best.pth`, a plain
  `model.state_dict()` loaded with `model.load_state_dict(...)`.
- **Training:** see [train_roi_classifier.py](#training-the-roi-classifier) and
  [docs/model_training_specs.md](docs/model_training_specs.md).

---

## Usage

### Requirements

```bash
pip install numpy torch torchvision ensemble-boxes pycocotools pillow tqdm
```

### Run the ensemble

The `--mode` flag selects the prediction set:

```bash
# Validation (metrics computed because a GT JSON is available)
python ensemble_detections_wbf_roi_classifier.py --mode val

# Test (no GT -> metrics skipped, classifier still runs)
python ensemble_detections_wbf_roi_classifier.py --mode test
```

Override the output path with `--output`:

```bash
python ensemble_detections_wbf_roi_classifier.py --mode test --output out/test_preds.json
```

The per-model prediction paths, weights, temperatures, IoU threshold and
classifier strategy are configured in the `if __name__ == "__main__"` block of
the script.

---

## Using the script with any number of models

The script is agnostic to the number and identity of the models. To use your own
models:

1. Save each model's detections as a COCO detection list (see format above) for
   the split you care about.
2. Add each path to the `val_json_paths` and/or `test_json_paths` lists.
3. Provide one `weight` and one `temperature` per model, in the same order.
   - `weights[i]` is the WBF fusion weight for model `i`.
   - `temperature_values[i]` is the temperature for model `i`.
4. Run with `--mode val` (to get metrics) or `--mode test`.

For example, to ensemble just two models:

```python
val_json_paths = [
    "data/predictions/val/ModelA.json",
    "data/predictions/val/ModelB.json",
]
weights = [1.0, 1.5]
temperature_values = [1.0, 0.7]
```

The order of `weights` and `temperature_values` must match the order of the
paths. Adding/removing a model only requires updating these three lists.

---

## Finetuning weights and temperatures

`finetune_weights_temperatures.py` performs a **greedy, coordinate-wise** search
over the ensemble hyper-parameters to maximize validation mAP:

```bash
python finetune_weights_temperatures.py \
    --predictions data/predictions/val/Strip-R-CNN.json \
                  data/predictions/val/DeIMv2.json \
                  ... \
    --gt data/gt/instances_val2017.json
```

It runs **two sequential phases**, looping over every model in order:

1. **Weights** — for each model, sweep `weight` from `0` to `5` in steps of `0.1`.
2. **Temperatures** — for each model, sweep `temperature` from `0.1` to `2.0`
   in steps of `0.1` (temperature must be strictly greater than 0, so the grid
   starts at 0.1).

Every candidate value that **increases** the validation mAP (COCO bbox
mAP @ `[0.5:0.95]`) is adopted. The final tuned `weights` and
`temperature_values` are written to `finetuned_params.json`, ready to paste
into the main ensemble script.

> The search is greedy (one parameter at a time), not a global grid search, so
> it is fast enough to run but not guaranteed to be globally optimal.

---

## Training the RoI classifier

`train_roi_classifier.py` reproduces `checkpoints/roi_classifier_best.pth`.
The procedure:

1. Build candidate boxes by fusing the per-model prediction files of a split
   with WBF (`--predictions ... ` plus `--weights`/`--temperatures`), or load a
   precomputed list of fused detections (`--candidates FILE`).
2. Label every candidate: a max IoU ≥ 0.5 with a ground-truth box makes it a
   true positive, otherwise it is a false positive.
3. Downsample the majority class so the two classes are balanced (We used
   a balanced set of 43,508 boxes from 363,609 candidates).
4. Split by image into training and validation sets, then fine-tune a
   torchvision ResNet-50 (`IMAGENET1K_V2`) truncated before avgpool/fc with
   the exact RoIAlign + MLP head the ensemble script uses, and save the best
   checkpoint.

```bash
python train_roi_classifier.py \
    --images /path/to/ClearSAR/data/images/train \
    --annotations data/gt/instances_train.json \
    --predictions data/predictions/train/Strip-R-CNN.json \
                  data/predictions/train/DeIMv2.json \
                  ... (all 13 files) \
    --weights 0.6,0.0,2.5,3.8,3.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 \
    --temperatures 1.5,0.9,0.6,1.0,0.6,1.3,1.5,0.9,0.9,0.6,0.6,0.6,0.6 \
    --output checkpoints/roi_classifier_best.pth
```

Training uses Adam with `lr 1e-4`, 50 epochs, one image per step (gradient
accumulation across `--batch-images` images) and no augmentation; the full
training configuration is in the script's docstring. Every run writes a sidecar
`<output>.summary.json` with the labeled box counts and the best validation
accuracy. The saved file is a plain state dict of the `RoIClassifier` defined
in `ensemble_detections_wbf_roi_classifier.py`, so it loads directly.

---

## Model training specifications

[docs/model_training_specs.md](docs/model_training_specs.md) documents, for
each of the 13 detectors behind the shipped prediction files, how it was
trained and how its predictions were produced: architecture, framework,
initial weights, training data, hyper-parameters, augmentation, checkpoint
selection and inference settings, following the methods described in the
paper.

---

## Models used for the shipped predictions

The bundled predictions were produced by 13 detectors spanning **10
architectures** (the D-FINE and RF-DETR families contribute several variants).
The index order below maps one-to-one onto the `weights` and
`temperature_values` lists in the main script. Per-detector validation mAPs are
reported in [docs/model_training_specs.md](docs/model_training_specs.md).

| Index | Prediction file             | Model / variant                         | Architecture                                           | Weight | Temp |
| ----- | --------------------------- | --------------------------------------- | ------------------------------------------------------ | ------ | ---- |
| 0     | `Strip-R-CNN.json`          | Strip R-CNN                             | Two-stage, StripNet backbone + oriented RPN (MMRotate) | 0.6    | 1.5  |
| 1     | `DeIMv2.json`               | DEIMv2-M                                | HGNetV2-B2 + hybrid encoder + DEIM decoder             | 0.0    | 0.9  |
| 2     | `RF_DETR.json`              | RF-DETR 2XLarge                         | DINOv2 backbone + deformable-attention decoder         | 2.5    | 0.6  |
| 3     | `YOLO.json`                 | YOLO11x                                 | Single-stage, DFL box head (Ultralytics)               | 3.8    | 1.0  |
| 4     | `DFine.json`                | D-FINE-M (ensemble of 3, flip TTA)      | HGNetV2-B2 + hybrid encoder, 4 decoder layers          | 3.0    | 0.6  |
| 5     | `CO-DETR.json`              | CO-DETR                                 | DETR-style + one-to-many auxiliary heads               | 0.0    | 1.3  |
| 6     | `DINO-DETR.json`            | DINO                                    | DETR-style + contrastive denoising                     | 0.0    | 1.5  |
| 7     | `RF_DETR_psudo.json`        | RF-DETR 2XLarge, Stage I pseudo-labels  | same as index 2 (smaller cls. layer)                   | 0.0    | 0.9  |
| 8     | `DDQ.json`                  | DDQ-DETR                                | DETR-style, class-agnostic NMS query selection         | 0.0    | 0.9  |
| 9     | `RTMDet.json`               | RTMDet                                  | One-stage anchor-free, CSPNeXt + PAFPN                 | 0.0    | 0.6  |
| 10    | `GLIP.json`                 | GLIP                                    | Vision-language, region–phrase grounding               | 0.0    | 0.6  |
| 11    | `DFine_exp08b.json`         | D-FINE-M (single, exp08b)               | same as index 4                                        | 0.0    | 0.6  |
| 12    | `RF_DETR_pseudolabels.json` | RF-DETR 2XLarge, Stage II pseudo-labels | same as index 2 (smaller cls. layer)                   | 0.0    | 0.6  |

The two RF-DETR pseudo-label variants (indices 7 and 12) keep the RF-DETR
2XLarge architecture (DINOv2 backbone, deformable decoder) but train it on
extended labels: **Stage I** adds high-confidence detections from RF-DETR and
YOLO11x on the training images, and **Stage II** adds 786 test images labeled
by an earlier 12-source version of this fusion. The full per-model training
specifications are in
[docs/model_training_specs.md](docs/model_training_specs.md).

The shipped tuned hyper-parameters (from the main script) are:

```python
weights           = [0.6, 0.0, 2.5, 3.8, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
temperature_values = [1.50, 0.90, 0.60, 1.00, 0.60, 1.30, 1.50, 0.90, 0.90, 0.60, 0.60, 0.60, 0.60]
```

A weight of `0.0` gives its source's boxes no influence on the fused geometry or
confidence, but they still join clusters and increment the agreement count `P`,
as described in the paper.
