# Model training specifications

This page documents, for each of the 13 detectors that produced the prediction
files shipped in `data/predictions/`, how it was trained and how its
predictions were generated. The specifications follow the methods described in
our paper, *From Diverse Detectors to Refined Predictions: A Multi-Evidence
Pipeline for Sentinel-1 SAR RFI Detection* (ClearSAR Track 1 Challenge, IEEE
ICIP 2026).

**Convention.** Each detector entry lists its architecture, framework, initial
weights, training data, training setup, inference settings, single-file
validation mAP, and the weight/temperature used in the ensemble. Training
details follow each framework's published recipe for the released model sizes
used here.

**Validation setup.** The team split the official ClearSAR training set into an
internal train/validation split: image ids shuffled with Python `random` and
seed 42, first 90 % (2,838 images, 8,256 boxes) for training and the remaining
316 images (1,032 boxes) for validation. All validation mAP numbers below are
COCO mAP @ [0.5:0.95] of each single prediction file on that 316-image
validation split, measured with pycocotools. The 786 test images have no public
labels.

The 13 prediction files (`data/predictions/val/*.json` and
`data/predictions/test/*.json`) are in a fixed order. The index below maps
one-to-one onto the `weights` and `temperature_values` lists in
`ensemble_detections_wbf_roi_classifier.py` and onto the slots of
`finetune_weights_temperatures.py`.

| Index | Prediction file | Model | Architecture (class) | Trained by | Validation mAP | Weight | Temperature |
|---|---|---|---|---|---|---|---|
| 0 | `Strip-R-CNN.json` | Strip R-CNN | Two-stage, StripNet backbone + oriented RPN | Ziad Fahmy | 0.2398 | 0.6 | 1.5 |
| 1 | `DeIMv2.json` | DEIMv2-M | HGNetV2-B2 + hybrid encoder + DEIM decoder | Abdelrahman Elnenaey | 0.3935 | 0.0 | 0.9 |
| 2 | `RF_DETR.json` | RF-DETR 2XLarge | DINOv2 backbone + deformable-decoder | Abdelrahman Elnenaey | 0.4235 | 2.5 | 0.6 |
| 3 | `YOLO.json` | YOLO11x | Single-stage, DFL box head | Youssif Abdelaziz | 0.4201 | 3.8 | 1.0 |
| 4 | `DFine.json` | D-FINE-M ensemble (3 models) | HGNetV2-B2 + hybrid encoder + 4-decoder | Youssif Abdelaziz | 0.4298 | 3.0 | 0.6 |
| 5 | `CO-DETR.json` | CO-DETR | DETR-style + one-to-many auxiliary heads | Ziad Fahmy | 0.3775 | 0.0 | 1.3 |
| 6 | `DINO-DETR.json` | DINO | DETR-style + contrastive denoising | Ziad Fahmy | 0.3694 | 0.0 | 1.5 |
| 7 | `RF_DETR_psudo.json` | RF-DETR 2XLarge (Stage I pseudo-labels) | DINOv2 backbone + deformable-decoder | Nour Eddine Hassan | 0.4263 | 0.0 | 0.9 |
| 8 | `DDQ.json` | DDQ-DETR | DETR-style, class-agnostic NMS-selected queries | Ziad Fahmy | 0.3946 | 0.0 | 0.9 |
| 9 | `RTMDet.json` | RTMDet | One-stage anchor-free, CSPNeXt + PAFPN | Ziad Fahmy | 0.3255 | 0.0 | 0.6 |
| 10 | `GLIP.json` | GLIP | Vision-language, region–phrase grounding | Ziad Fahmy | 0.3804 | 0.0 | 0.6 |
| 11 | `DFine_exp08b.json` | D-FINE-M, single model | HGNetV2-B2 + hybrid encoder + 4-decoder | Youssif Abdelaziz | 0.4384 | 0.0 | 0.6 |
| 12 | `RF_DETR_pseudolabels.json` | RF-DETR 2XLarge (Stage II pseudo-labels) | DINOv2 backbone + deformable-decoder | Nour Eddine Hassan | 0.4309 | 0.0 | 0.6 |

A weight of `0.0` still feeds the box **shape** into the fusion; only the score
contribution is zeroed. All detectors were trained on the single RFI class.

---

## Sources with a non-zero fusion weight

These four sources carry all of the fusion weight.

### YOLO11x (`YOLO.json`) — index 3

YOLO11x is a single-stage detector from Ultralytics. It carries the largest
weight in the fusion.

- **Architecture:** YOLO11x, whose box head uses distribution focal loss (DFL).
  About 56.9 million parameters. One output class (RFI).
- **Framework:** Ultralytics 8.4.19 on PyTorch 2.5.1 (CUDA 12.1 build),
  Python 3.10.
- **Initial weights:** COCO-pretrained YOLO11x (`yolo11x.pt`) from Ultralytics,
  with the output layer set to one class.
- **Training data:** the internal training split, 2,838 images. COCO
  annotations converted to YOLO text labels with a single class. No
  pseudo-labels, test images or external data.
- **Training setup:**
  - Input size 1600 px, 120 epochs, one NVIDIA V100 32 GB GPU, about 19 hours.
  - Batch size 2 with gradient accumulation to a nominal batch of 64.
  - SGD with learning rate 0.01, momentum 0.937, weight decay 0.0005. Linear
    decay to 1 % of the initial rate with 3 warm-up epochs. Mixed precision,
    seed 42.
  - Loss gains: box 10.0, classification 0.5, DFL 2.0 (higher than the
    standard Ultralytics gains of 7.5 and 1.5).
  - Augmentation: mosaic (off for the last 10 epochs), horizontal flip
    (p = 0.5), saturation and brightness jitter (0.3 each), translation (0.1),
    scale (0.5). No rotation, vertical flip, mixup or copy-paste. The standard
    Ultralytics light augmentations (blur, median blur, grayscale, CLAHE) were
    on, each with p = 0.01.
  - **Checkpoint selection:** the epoch with the best validation
    mAP@[0.5:0.95] in the Ultralytics validator — epoch 116 (1-based).
- **Inference:**
  - Each image is letterboxed to 1600 px.
  - Ultralytics built-in test-time augmentation runs three passes: full scale,
    0.83x with horizontal flip, and 0.67x scale. Outputs of the three passes
    are combined and one NMS at IoU 0.6 is run.
  - Confidence threshold 0.001, at most 100 boxes per image.
  - Boxes written in COCO `[x, y, w, h]` with category 1; coordinates rounded
    to two decimals, scores to six; zero-area boxes dropped.
  - **Validation mAP:** 0.4201 (AP50 0.6812, AP75 0.4576). Without TTA the
    same checkpoint scores 0.4068.
- **Fusion:** weight 3.8, temperature 1.0 (index 3).

### D-FINE-M ensemble (`DFine.json`) — index 4

D-FINE is a DETR-style detector that predicts each box edge as a probability
distribution and refines it through the decoder layers. This source merges
three D-FINE-M models, each run on the original image and on two flipped
copies.

| Model | Initial weights | Train size (multi-scale range) | Epochs (plain final phase) | Batch | LR, head / backbone | Schedule | Validation mAP (training log) | Weight in ensemble |
|---|---|---|---|---|---|---|---|---|
| Model 1 | ImageNet HGNetV2-B2 | 640 (480–800) | 132 (last 12) | 8 | 6.25e-5 / 6.25e-6 | 500-iter linear warm-up, then constant | 0.4007 | 0.4007 |
| Model 2 | D-FINE-M, Objects365 + COCO | 800 (576–992) | 72 (last 12) | 8 | 1.25e-4 / 1.25e-5 | constant | 0.4177 | 0.4177 |
| Model 3 | D-FINE-M, Objects365 + COCO | 800 (576–992) | 120 (last 20) | 16 | 1.77e-4 / 1.77e-5 | cosine decay to 1e-6 | 0.4106 | 0.4110 |

- **Architecture:** D-FINE-M. HGNetV2-B2 backbone, hybrid encoder, 4 decoder
  layers, 300 object queries, one class (RFI). About 19.2 million parameters.
- **Framework:** the official D-FINE PyTorch code ([D-FINE](https://github.com/Peterande/D-FINE),
  commit `d669475`) with a ClearSAR dataset config and a vertical-flip
  transform added. PyTorch 2.3.1, torchvision 0.18.1. Passes merged with
  ensemble-boxes 1.0.9.
- **Initial weights:** Model 1 from the ImageNet-pretrained HGNetV2-B2
  backbone shipped with D-FINE, detection head from scratch. Models 2 and 3
  from the official D-FINE-M checkpoint pretrained on Objects365 + COCO
  (`dfine_m_obj2coco.pth`), classification head re-initialized for one class.
- **Training data:** the internal training split, 2,838 images, 8,256 boxes.
- **Training setup (shared):** AdamW, weight decay 1.25e-4, gradient clipping
  at 0.1, EMA decay 0.9999, mixed precision, seed 0, one V100 32 GB GPU.
  Augmentation: random photometric distortion, zoom-out, IoU crop, horizontal
  flip, square resize, per-batch multi-scale in 32-px steps; models 2 and 3
  also use vertical flips. Two-stage schedule: in the final phase (epochs in
  parentheses) distortion, zoom-out, IoU crop and multi-scale are off; training
  restarts from the best first-phase checkpoint after every non-improving
  epoch. Kept checkpoint: best EMA validation mAP of the final phase.
- **Inference:** each model at its square size (640 or 800); TTA with original,
  horizontal flip and vertical flip (nine passes total); 300 top-scoring
  queries per pass, scores below 0.02 dropped, NMS at IoU 0.6, at most 50
  boxes; passes merged with WBF (IoU 0.65, skip 0.001), each pass weighted by
  its model's ensemble weight; 300 highest-scoring fused boxes kept.
- **Validation mAP:** 0.4298 (AP50 0.7146, AP75 0.4477).
- **Fusion:** weight 3.0, temperature 0.6 (index 4).

### RF-DETR 2XLarge (`RF_DETR.json`) — index 2

RF-DETR is a DETR-style detector from Roboflow pairing a DINOv2 vision
transformer backbone with a deformable-attention decoder.

- **Architecture:** RF-DETR 2XLarge. DINOv2 backbone with 12 transformer
  layers, 768 channels, 20-pixel patches; 5-layer decoder with deformable
  cross-attention, hidden size 512, two-stage query selection; 300 object
  queries (13 query groups during training). About 127 million parameters.
- **Framework:** Roboflow `rfdetr` with the `rfdetr-plus` extension,
  trained through PyTorch Lightning 2.6.1.
- **Initial weights:** COCO-pretrained RF-DETR 2XLarge (`rf-detr-xxlarge.pth`).
- **Training data:** internal training split (2,838 images, 8,256 boxes),
  single RFI class; validation split evaluated after every epoch.
- **Training setup:**
  - Square resize with multi-scale training (expanded scales) enabled.
  - Effective batch size 16 (batch 4, 4 gradient-accumulation steps).
  - Base learning rate 1e-4, kept constant. Different rates for backbone and
    decoder via `lr_encoder` 1.5e-4, `lr_vit_layer_decay` 0.8,
    `lr_component_decay` 0.7.
  - Weight decay 1e-4, gradient clipping at 0.1, no warm-up.
  - EMA decay 0.993.
  - 20 epochs; the kept checkpoint holds the EMA weights of epoch 9
    (0-based), the best EMA validation mAP.
- **Inference:** 300 entries per image, category 1, COCO `[x, y, w, h]`,
  scores down to about 0.003.
- **Validation mAP:** 0.4235 (AP50 0.6991, AP75 0.4458).
- **Fusion:** weight 2.5, temperature 0.6 (index 2).

### Strip R-CNN (`Strip-R-CNN.json`) — index 0

Strip R-CNN is a two-stage detector designed for objects with a high aspect
ratio in remote sensing images. ClearSAR interference mostly appears as thin
horizontal stripes (median box 138 × 10 px in the official training set).

- **Architecture:** Strip R-CNN (Yuan et al., arXiv:2501.03775). A StripNet
  backbone built from large horizontal and vertical strip convolutions, an FPN
  neck, an oriented region proposal network and a RoI head.
- **Framework:** MMRotate (OpenMMLab).
- **Training setup:** following the reference recipe shipped with the official
  Strip R-CNN code (an MMRotate 1x DOTA-style schedule):
  - 12 epochs, batch size 2 per GPU, one class (RFI).
  - SGD with learning rate 0.005, momentum 0.9, weight decay 0.0001, linear
    warm-up for the first epoch; horizontal flip with p = 0.5 and random
    rotation (p = 0.5) augmentation.
  - Checkpoint selection: the epoch with the best validation mAP.
- **Inference:**
  - Predictions produced with the MMRotate test tool; every rotated box
    converted to the smallest axis-aligned box containing it, written in COCO
    format, category 1.
  - Boxes extending past the left/top edge are clipped at that edge.
  - Validation file: 612 boxes on 271 of 316 images, all scores ≥ 0.9.
    Test file: 2,839 boxes on 770 of 786 images, all scores ≥ 0.5.
- **Validation mAP:** 0.2398 (AP50 0.4150, AP75 0.2597).
- **Fusion:** weight 0.6, temperature 1.5 (index 0).

---

## Other sources (weight 0.0)

These sources carry weight 0.0 but are still passed to the fusion, as in the
submitted run; their boxes still contribute to fused box shapes.

### D-FINE-M exp08b (`DFine_exp08b.json`) — index 11

A single D-FINE-M model and the best single source on the validation split
(0.4384). Compared with the ensemble models, its backbone learns at the same
rate as the head and it trains on a longer 200-epoch schedule.

- **Architecture:** D-FINE-M, as above (HGNetV2-B2, hybrid encoder, 4 decoder
  layers, 300 queries).
- **Framework:** the same D-FINE code as the ensemble, PyTorch 2.2.2 /
  torchvision 0.17.2.
- **Initial weights:** `dfine_m_obj2coco.pth`, classification head
  re-initialized for one class.
- **Training data:** internal training split (2,838 images, 8,256 boxes).
- **Training setup:**
  - Input 800×800 with per-batch multi-scale 576–992.
  - Augmentation: photometric distortion, zoom-out, IoU crop, horizontal and
    vertical flips.
  - 200 epochs, batch size 8; the last 33 epochs run without distortion,
    zoom-out, IoU crop and multi-scale.
  - AdamW, constant LR 1.25e-4 for head **and** backbone, weight decay
    1.25e-4, gradient clipping at 0.1, EMA decay 0.9999, full precision
    (FP32), seed 42, one V100 32 GB, about 15.7 hours.
  - Same final-phase checkpoint selection as the ensemble.
- **Inference:** one pass at 800×800, no TTA, no score threshold; D-FINE
  postprocessor returns the 300 top-scoring queries per image with sigmoid
  scores; boxes mapped back to original size.
- **Validation mAP:** 0.4384 (AP50 0.7083, AP75 0.4670).
- **Fusion:** weight 0.0, temperature 0.6 (index 11).

### DEIMv2-M (`DeIMv2.json`) — index 1

DEIMv2 is a real-time DETR-style detector predicting a fixed set of boxes
directly, without anchors or NMS, using the same HGNetV2-B2 backbone as
D-FINE-M with its own decoder and recipe.

- **Architecture:** DEIMv2-M. HGNetV2-B2 backbone, hybrid encoder (hidden 256,
  one transformer encoder layer), 4-layer DEIM decoder with 300 queries.
- **Framework:** the official DEIMv2 code ([DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)).
- **Initial weights:** ImageNet-pretrained HGNetV2-B2 (`PPHGNetV2_B2_stage1.pth`,
  the default backbone weights of the DEIMv2 code); encoder/decoder from random
  init. No COCO-pretrained detector weights.
- **Training data:** internal training split (2,838 images, 8,256 boxes). Every
  training and validation image was preprocessed with CLAHE: OpenCV CLAHE on
  the lightness channel of LAB, clip limit 2.0, 32×32 tiles, color channels
  unchanged. No pseudo-labels, test images or external data.
- **Training setup:**
  - 132 epochs, batch size 32, mixed precision, seed 0.
  - Input 640×640; up to epoch 89 batches resized to a random scale 480–800.
  - AdamW, LR 4e-4 (4e-5 backbone), weight decay 1e-4; 2,000 warm-up
    iterations, constant until epoch 49, then cosine decay to half the base
    rate; EMA decay 0.9999.
  - Augmentation: horizontal flip throughout; epochs 4–89 also photometric
    distortion, zoom-out, IoU crop, copy-blend; epochs 4–48 also mosaic and
    mixup; last 42 epochs without the stronger augmentations.
  - Loss/matching: DEIMv2 defaults (Hungarian matching, matchability-aware
    classification loss, L1 + GIoU, fine-grained localization losses).
  - Checkpoint selection: after the stronger augmentations stop, a checkpoint
    is saved whenever validation mAP improves; otherwise training reverts to
    the best first-phase checkpoint. Kept: epoch 94 (0-based), 0.4009 mAP on
    CLAHE-enhanced validation images; its EMA weights produced both files.
- **Inference:** runs on the original PNGs (no CLAHE), resized to 640×640
  (aspect not kept), pixels scaled to [0, 1]; one forward pass, no TTA; keeps
  the 300 highest-scoring entries per image (sigmoid, no NMS), threshold 1e-6.
- **Validation mAP:** 0.3935 (AP50 0.6469, AP75 0.4199).
- **Fusion:** weight 0.0, temperature 0.9 (index 1).

### CO-DETR, DINO, DDQ-DETR, RTMDet and GLIP (indexes 5, 6, 8, 9, 10)

Ziad Fahmy trained these five detectors, all with a single RFI class. Each
entry describes the training recipe of its framework/reference for the released
model sizes used in this work.

- **(index 5) CO-DETR** (Zong et al., ICCV 2023) — DETR-style detector trained
  with extra auxiliary heads using one-to-many label assignment, giving the
  encoder denser supervision (auxiliary heads used only in training).
  - Framework: official [Co-DETR](https://github.com/Sense-X/Co-DETR)
    (mmdetection-based).
  - Training setup: 12 epochs; batch size 16; AdamW LR 1e-4 (backbone 1e-5),
    weight decay 1e-4, gradient clipping at 0.1; multi-scale resize 480–800,
    random crop and flip augmentation.
  - Inference characteristics: 106–263 boxes per image (~180 avg), scores
    0.001–0.89; 57,255 validation / 144,872 test boxes.
  - Validation mAP 0.3775 (AP50 0.6435, AP75 0.3861); weight 0.0, temp 1.3.

- **(index 6) DINO** (Zhang et al., ICLR 2023) — DETR-style detector with
  contrastive denoising training, mixed query selection and look-forward-twice
  box refinement.
  - Framework: official mmdetection [DINO](https://github.com/open-mmlab/mmdetection)
    configs.
  - Training setup: 12 epochs; batch size 16; AdamW LR 1e-4, weight decay 1e-4;
    multi-scale resize 480–800, random crop and flip augmentation.
  - Inference characteristics: 300 boxes per image, scores 0.007–0.94;
    94,800 / 235,800 boxes.
  - Validation mAP 0.3694 (AP50 0.6343, AP75 0.3769); weight 0.0, temp 1.5.

- **(index 8) DDQ-DETR** (Zhang et al., CVPR 2023) — starts from a dense set of
  queries, keeps distinct ones with class-agnostic NMS, refines with one-to-one
  matching.
  - Framework: official mmdetection [DDQ-DETR](https://github.com/open-mmlab/mmdetection)
    configs.
  - Training setup: 12 epochs; batch size 16; AdamW LR 1e-4, weight decay 1e-4;
    multi-scale resize 480–800, random crop and flip augmentation.
  - Inference characteristics: 300 boxes per image, scores 0.006–0.92;
    94,800 / 235,800 boxes.
  - Validation mAP 0.3946 (AP50 0.6740, AP75 0.4206); weight 0.0, temp 0.9.

- **(index 9) RTMDet** (Lyu et al., 2022) — one-stage, anchor-free CNN detector
  for real-time use; CSPNeXt backbone, CSPNeXt PAFPN neck, dynamic soft label
  assignment.
  - Framework: official [RTMDet](https://github.com/open-mmlab/mmdetection)
    configs.
  - Training setup: input 640; 300 epochs; batch size 256; SGD LR 0.004,
    momentum 0.9, weight decay 0.05; EMA decay 0.0002; multi-scale resize
    480–960 and random flip/crop augmentation.
  - Inference characteristics: 300 boxes per image, scores 0.011–0.64;
    94,800 / 235,800 boxes.
  - Validation mAP 0.3255 (AP50 0.6033, AP75 0.3208); weight 0.0, temp 0.6.

- **(index 10) GLIP** (Li et al., CVPR 2022) — vision-language detector that
  treats detection as phrase grounding, scoring image regions against the words
  of a text prompt.
  - Framework: official [GLIP](https://github.com/microsoft/GLIP).
  - Training setup: start from the published GLIP-L (Swin-L) checkpoint and
    fine-tune with AdamW (visual backbone LR ≈ 1e-5, text/head LR ≈ 2e-5) for
    the epochs needed on the target set; detect with the text prompt "RFI".
  - Inference characteristics: 1–100 boxes per image (~40 avg), scores
    0.069–0.95; 12,691 / 30,985 boxes.
  - Validation mAP 0.3804 (AP50 0.6538, AP75 0.3973); weight 0.0, temp 0.6.

### RF-DETR 2XLarge, Stage I pseudo-labels (`RF_DETR_psudo.json`) — index 7

Trained on the training labels extended with Stage I pseudo-labels (see
[Pseudo-labeling](#pseudo-labeling)).

- **Architecture:** same RF-DETR 2XLarge as `RF_DETR.json`, with a smaller
  classification layer; about 126 million parameters.
- **Framework:** Roboflow `rfdetr`, class `RFDETR2XLarge`.
- **Initial weights:** COCO-pretrained `rf-detr-xxlarge.pth`.
- **Training data:** 2,838 internal training images with 10,232 boxes (8,256
  original + 1,976 Stage I pseudo-label boxes).
- **Training setup:**
  - Input resized to 1000×1000 with multi-scale training.
  - Batch 4 with 4 gradient-accumulation steps (effective batch 16).
  - Same learning rates as `RF_DETR.json`, constant; weight decay 1e-4,
    gradient clipping 0.1, mixed precision, seed 42, one GPU.
  - EMA decay 0.993, evaluated on validation after every epoch; best checkpoint
    kept: epoch 8 (0-based).
- **Inference:** one pass at 1000×1000, no TTA, score threshold 0.001, no extra
  NMS; 300 entries per image. Inference run by Abdelrahman Elnenaey.
- **Validation mAP:** 0.4263 (AP50 0.6929, AP75 0.4551).
- **Fusion:** weight 0.0, temperature 0.9 (index 7).

### RF-DETR 2XLarge, Stage II pseudo-labels (`RF_DETR_pseudolabels.json`) — index 12

Same architecture and recipe as the Stage I model; the training set adds the
786 test images labeled with the output of an earlier version of this fusion.

- **Architecture / framework:** as the Stage I model.
- **Initial weights:** COCO-pretrained `rf-detr-xxlarge.pth` (not the Stage I
  weights).
- **Training data:** 3,624 images with 22,700 boxes (2,838 training images with
  8,256 original boxes, plus 786 test images with 14,444 Stage II pseudo-label
  boxes). Validation split held out; no test annotations used.
- **Training setup:** identical to the Stage I model; best EMA checkpoint
  kept: epoch 5 (0-based).
- **Inference:** 100 boxes per image. Inference run by Nour Eddine Hassan.
- **Validation mAP:** 0.4309 (AP50 0.7109, AP75 0.4635).
- **Fusion:** weight 0.0, temperature 0.6 (index 12).

---

## Pseudo-labeling

The models behind indexes 7 and 12 were trained with the same detector and
recipe; only the training data differ.

### Stage I (refined training labels)

1. Two detectors trained on the original labels were run on the 2,838 training
   images: an RF-DETR 2XLarge and the YOLO11x behind `YOLO.json` (1600 px, TTA).
2. Score filters: RF-DETR boxes ≥ 0.5 (6,112 boxes); YOLO11x boxes ≥ 0.3
   (7,957 of 99,548 boxes).
3. Merge: predicted boxes matched one-to-one with existing labels (IoU ≥ 0.5);
   unmatched predictions added, original boxes kept. RF-DETR matched first,
   then YOLO11x against the originals + added RF-DETR boxes.
4. Result: 10,232 boxes on 2,838 images (8,256 original + 804 RF-DETR + 1,172
   YOLO11x), new boxes on 1,065 images.

### Stage II (test-set pseudo-labels)

1. An earlier 12-source version of this fusion (all sources except index 12,
   same weights/temperatures, IoU 0.7, skip 0.06, no rescoring) was run on the
   786 test images.
2. 14,444 fused boxes were used as labels, with no score threshold.
3. Test images + labels were added to the training set (3,624 images, 22,700
   boxes); validation stayed held out.
4. RF-DETR 2XLarge retrained from the COCO weights with the recipe above.

### Validation results (with the same recipe, differing only in training data)

| Training labels | Images | Boxes | Best EMA validation mAP (training log) | Best epoch |
|---|---|---|---|---|
| Original labels (reference, not a source) | 2,838 | 8,256 | 0.4159 | 5 |
| Stage I (index 7) | 2,838 | 10,232 | 0.4274 | 8 |
| Stage II (index 12) | 3,624 | 22,700 | 0.4326 | 5 |

---

## RoI classifier (rescoring model)

`checkpoints/roi_classifier_best.pth` is a binary encoder that scores every
fused box with the probability that it is a correct detection (IoU ≥ 0.5 with a
ground-truth box). It is a ResNet-50 backbone truncated before avgpool/fc plus
RoIAlign pooling and a small MLP head. Training follows the paper's
§ BBox Confidence Rescorer; the full procedure and the training configuration
are implemented in `train_roi_classifier.py`.

- **Architecture:**
  1. Input: the full image at original size (Pillow, RGB), pixels scaled to
     [0, 1], normalized with ImageNet mean (0.485, 0.456, 0.406) and std
     (0.229, 0.224, 0.225). The image is not resized or padded; one image is
     processed at a time with all of its boxes.
  2. Backbone: torchvision ResNet-50 up to and including `layer4` (no avgpool,
     no fc) — 2048 channels at stride 32.
  3. RoIAlign: `torchvision.ops.roi_align` on each fused box `[x1, y1, x2, y2]`
     in pixels, output size 7×7, spatial scale 1/32, `aligned=True`.
  4. Head: adaptive average pooling to a 2048-d vector, Linear 2048→512, ReLU,
     Dropout 0.5, Linear 512→1 logit.
  5. Output: sigmoid → probability. About 24.6 million parameters.
- **Data building (as described):** inference on the training set gave 363,609
  candidate boxes; a candidate with IoU ≥ 0.5 to a ground-truth box is a true
  positive, all others false positives; false positives were downsampled to
  give a balanced set of 43,508 boxes.
- **Fine-tuning:** starts from torchvision's ImageNet-pretrained ResNet-50
  (`IMAGENET1K_V2`) with a new head; the whole network (backbone + head) is
  fine-tuned.
- **Checkpoint format:** a raw `model.state_dict()` of the `RoIClassifier`
  defined in `ensemble_detections_wbf_roi_classifier.py` (keys `backbone.*`
  and `head.*`), loaded directly with `model.load_state_dict(...)`.