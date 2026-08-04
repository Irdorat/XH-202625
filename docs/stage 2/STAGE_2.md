# STAGE 2

## Team Responsibilities

| Member | Task | Main files | Expected outcome |
|---|---|---|---|
| Xu Ruiqin | Task 1 | `src/ml_comp/data.py`<br>`scripts/prepare_data.py`<br>`data/splits/baseline_v2/` | Scene-aware split, rare-class statistics, and a validated 10k scene builder |
| Ma Qingfeng | Task 2 | `src/ml_comp/tiling.py`<br>`src/ml_comp/inference.py`<br>`scripts/predict.py` | End-to-end 10k inference pipeline with slicing, merging, COCO JSON output, visualization, and runtime measurements |
| Geng Ziyao | Task 3 | `src/ml_comp/augmentations.py`<br>`configs/baseline_v2/copy_paste.yaml` | Tested Copy-Paste module with valid annotations, class statistics, and visual examples |
| Wu Hang | Task 4 | `src/ml_comp/scene_mixing.py`<br>`src/ml_comp/sampler.py`<br>`configs/baseline_v2/scene_mixing.yaml`<br>`configs/baseline_v2/fair_training.yaml` | Tested Scene Mixing module and balanced training sampler |
| Pavel Klykov | Task 5 | `scripts/train.py`<br>`scripts/evaluate.py`<br>`src/ml_comp/metrics.py`<br>`configs/baseline_v2/experiments/` | Selected baseline v2 architecture, unified evaluator, experiment configurations, and control checkpoint |

---

## Task 1 — Source Dataset Improvement

**Owner:** Xu Ruiqin

### Goal

Analyze the dataset, identify rare classes, reconstruct or synthetically assemble source tiles into 10k×10k scenes, and create a fixed scene-aware split for baseline v2.

### Deliverables

- Dataset report containing image and object counts for all 25 classes, with rare classes identified.
- Scene-grouping and tile-position reconstruction method based on filenames, metadata, and visual boundary matching.
- Exact large-scene reconstruction when tile positions can be reliably determined.
- Synthetic 10k×10k scene assembly when exact reconstruction is not possible.
- New annotations in global scene coordinates; original `.txt` files must remain unchanged.
- Metadata for every tile: source file, scene coordinates, scale, and padding.
- Fixed scene-aware train/validation split under `data/splits/baseline_v2/`.
- Visual validation samples for assembled scenes and annotations.

### Acceptance Criteria

- Tiles from the same source scene never appear in both training and validation sets.
- The split preserves an acceptable distribution of rare and frequent classes in both subsets.
- All bounding boxes are correctly transformed into global coordinates.
- No objects are lost or duplicated during scene assembly.
- Exact reconstruction is used only when tile positions are reliable; otherwise, the scene is explicitly marked as synthetic.
- Re-running the preparation process produces the same fixed split.

### Deadlines and Dependencies

- Rare-class statistics: **August 5–6**. This blocks Tasks 3 and 4.
- Fixed scene-aware split: **August 7–8**. This is required by Tasks 2–5.

---

## Task 2 — Inference Pipeline for 10k×10k Images

**Owner:** Ma Qingfeng

### Goal

Build a model-independent, end-to-end inference pipeline for 10k×10k images using overlapping tiles and reliable global prediction merging.

### Deliverables

- Configurable slicing into overlapping 1024×1024 tiles, including correct right and bottom boundary handling.
- Global-coordinate metadata for every tile.
- Batched FP16 inference through a common predictor interface.
- Conversion of tile-local predictions into global image coordinates.
- Duplicate-detection merging with class-aware NMS.
- Common comparison interface for NMS, Soft-NMS, and WBF.
- Final predictions in COCO JSON format.
- Full-image visualization with final bounding boxes.
- Runtime report covering slicing, preprocessing, inference, merging, and total wall-clock time.

### Acceptance Criteria

- Tile size and overlap are configurable.
- Final bounding boxes are clipped to the original image boundaries.
- Duplicate detections in overlapping regions are merged correctly.
- Objects near tile boundaries are not lost.
- The pipeline can be tested with a mock or alternative predictor and does not depend on the final model.
- A complete run produces valid COCO JSON, visualization, and timing results.

### Deadlines and Dependencies

- Depends on the 10k scene format and fixed validation split from Task 1.
- Does not depend on completion of the final model selection in Task 5.

---

## Task 3 — Copy-Paste and Data Multiplication

**Owner:** Geng Ziyao

### Goal

Implement reproducible Copy-Paste augmentation that increases rare-class representation without contaminating the validation set.

### Deliverables

- Index of rare-class donor objects from the training set.
- Copy-Paste module with configurable scaling, flipping, overlap control, and placement limits.
- Controls for minimum and maximum object scale, pasted objects per image, source-object reuse, and real-to-synthetic object ratio.
- Configuration at `configs/baseline_v2/copy_paste.yaml`.
- Visual review set containing at least 100 synthetic images.
- Class-distribution report before and after Data Multiplication.

### Acceptance Criteria

- Only training images are used as donor sources; validation images are never used.
- Class labels and bounding boxes remain correct after every transformation.
- Placement respects the configured overlap and quantity limits.
- Reuse of the same source object does not exceed the configured frequency.
- A fixed seed produces reproducible results.
- The before/after report demonstrates the effect on rare-class representation.

### Deadlines and Dependencies

- Depends on rare-class definitions and the fixed split from Task 1.

---

## Task 4 — Scene Mixing and Fair Training

**Owner:** Wu Hang

### Goal

Implement reproducible Scene Mixing and balanced sampling to expose the model more frequently to rare classes while preserving controlled training behavior.

### Deliverables

- Scene Mixing module that combines 6–9 training images into a 1024×1024 sample.
- Correct transformation, clipping, and filtering of bounding boxes.
- Optional guarantee that each mixed sample contains at least one rare class.
- Visual validation set for Scene Mixing results.
- Fair Training sampler that derives each image weight from all classes present in that image.
- Weak, medium, and strong balancing modes.
- Configurations at `configs/baseline_v2/scene_mixing.yaml` and `configs/baseline_v2/fair_training.yaml`.
- Report comparing per-class sampling frequency before and after Fair Training.

### Acceptance Criteria

- Scene Mixing and Fair Training use training images only.
- Bounding boxes are correctly scaled, transformed, and clipped to their image regions.
- A box is removed when its retained object area falls below the configured threshold.
- The sampler limits the maximum repetition frequency of each image.
- Epoch length remains fixed in every balancing mode.
- Fixed seeds make both Scene Mixing and sampling reproducible.
- The frequency report clearly shows the effect of each balancing mode.

### Deadlines and Dependencies

- Depends on rare-class definitions and the fixed split from Task 1.

---

## Task 5 — Evaluation and Result Analysis

**Owner:** Pavel Klykov

### Goal

Create a unified and reproducible training/evaluation workflow, select the primary baseline v2 architecture, and prepare the control and ablation experiments.

### Deliverables

- Unified training script and unified evaluation script.
- Metrics implementation for Recall, Precision, FAR, F1, mAP50, mAP50–95, and per-class results.
- Confidence-threshold sweep.
- Documented evaluation protocol covering confidence threshold, matching IoU, NMS IoU, and class-aware one-to-one matching.
- Evaluation of available YOLO11n and YOLO26n checkpoints on the fixed validation set.
- RT-DETR-L retrained on the complete training set without `fraction=0.5`.
- Comparison of accuracy, speed, VRAM usage, and rare-class Recall.
- Selection of the primary baseline v2 architecture.
- `v2_control` checkpoint trained without Copy-Paste, Scene Mixing, or Fair Training.
- Experiment configurations for `v2_control`, `v2_copy_paste`, `v2_scene_mix`, `v2_fair_train`, and `v2_combo`.
- Model-comparison table and ablation-table template.

### Acceptance Criteria

- Every model is evaluated on the same fixed validation split with the same evaluator and matching rules.
- Every run stores its configuration, seed, split, metrics, and checkpoint.
- Model selection is supported by the comparison table, including rare-class Recall and resource usage.
- `v2_control` contains none of the three experimental data/training techniques.
- Experiment configurations differ only in the intended intervention wherever practical.
- The ablation template is ready for pipeline integration after **August 12**.

### Deadlines and Dependencies

- Evaluation depends on the fixed validation split from Task 1.
- Ablation experiments depend on the completed modules and configurations from Tasks 3 and 4.
- Pipeline integration begins after **August 12**.

---


## Stage-Level Definition of Done

Stage 2 is complete when:

- The fixed scene-aware split and all shared dataset artifacts are versioned and reproducible.
- The 10k inference pipeline passes boundary and duplicate-merging checks.
- Copy-Paste, Scene Mixing, and Fair Training pass annotation and reproducibility checks.
- All models use the same documented evaluation protocol.
- The primary baseline v2 architecture and `v2_control` checkpoint are selected and saved.
- Required configurations, metrics, checkpoints, reports, and visual validation artifacts are available for review.
