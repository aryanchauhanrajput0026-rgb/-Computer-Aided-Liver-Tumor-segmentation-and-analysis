# Liver Tumor Segmentation and Analysis

Educational research prototype for 3D CT liver and tumor segmentation with a 2D U-Net. This code does not provide medical diagnosis and should not be used for clinical decisions.

## Dataset and project status

This repository is configured for the Medical Segmentation Decathlon Task03 Liver dataset in the real filesystem location:

```powershell
data\Task03_Liver\Task_liver_dataset\Task03_Liver\Task03_Liver
```

The dataset contains:

- `imagesTr`: 131 training CT volumes
- `labelsTr`: 131 matching training masks
- `imagesTs`: 70 held-out test images
- `dataset.json`: defines labels `0 = background`, `1 = liver`, `2 = cancer`

The project keeps the legacy dataset and the original checkpoint untouched. New training outputs are saved under a distinct checkpoint filename and the app checks for the newest valid checkpoint before inference.

## Validate the dataset

From the project root:

```powershell
.\.venv\Scripts\python -m src.inspect_dataset --root "data\Task03_Liver\Task_liver_dataset\Task03_Liver\Task03_Liver" --json-report "outputs\msd_task03_dataset_inspection.json" --markdown-report "outputs\msd_task03_dataset_inspection.md"
```

The inspector ignores macOS metadata files such as `._*.nii.gz`, validates each image-label pair, verifies shapes and affine matrices, and ensures mask values stay within the verified set `{0, 1, 2}`.

## Validate and create reproducible patient-level splits

The Task03 split command validates the manifest, pair names, image/mask dimensions and affines, and every training mask before writing a two-way patient-level split. `imagesTs` is recorded only as external holdout image metadata; it is never added to train or validation:

```powershell
.\.venv\Scripts\python -m src.make_splits --dataset-root "data\Task03_Liver\Task_liver_dataset\Task03_Liver\Task03_Liver" --out data\msd_task03_liver_splits.json --seed 42 --validation-fraction 0.15 --confirm-semantic-labels
```

The generated split contains 112 training cases, 19 validation cases, and 70 separate external test images for this dataset. Existing `data\splits.json` and the original checkpoint are not modified.

## Environment and dependencies

Use the existing project environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

This project is CPU-friendly and can run on a laptop without CUDA; use conservative batch sizes when memory is limited.

## Train the model

The CPU loader reads one compressed case at a time and groups batches by patient to avoid repeatedly decompressing NIfTI slices. The laptop-safe configuration samples up to four slices per patient (prioritizing three tumor-positive slices when available), uses 256x256 tumor-centered training patches, defaults to equal cross-entropy class weights, and excludes classes absent from a batch target from its Dice-loss term. Class weights can be changed with `--class-weights`; training history records class-wise Dice, and checkpoint selection prioritizes sampled validation tumor Dice, with validation loss as a tie-breaker. These sampled metrics are not full-volume performance claims:

```powershell
.\.venv\Scripts\python -m src.train --splits data\msd_task03_liver_splits.json --epochs 1 --batch-size 1 --slices-per-case 4 --patch-size 256 --workers 0 --lr 0.001 --checkpoint checkpoints\best_model_msd.pt
```

Training writes status to `outputs\msd_task03_training_status.json`, loss history to `outputs\msd_task03_training_history.csv`, and the best validation checkpoint to the selected checkpoint path. The original `outputs\training_history.csv` is preserved.

### Tumor-class diagnostic status

One-epoch fine-tuning diagnostics were run from `checkpoints\best_model_msd_extended.pt` with class weights `0.1/1/8` and `0.1/1/32`. A third run also excluded target-absent classes from the Dice loss. None predicted tumor voxels on sampled validation slices or full-volume `liver_74`; liver Dice fell below the extended checkpoint's 0.8047. These diagnostic checkpoints are experimental and are not selected by the app. Tumor segmentation remains unresolved.

The trainer also supports an optional tumor-logit margin term. On tumor target pixels it penalizes cases where the tumor logit is not at least the configured margin above the stronger background/liver logit. Its default weight is zero, so existing training behavior is unchanged unless explicitly enabled. A one-epoch diagnostic initialized from the extended checkpoint used margin weight `0.1`, margin `1.0`, learning rate `0.0001`, and the existing four-slices-per-case / 256x256 tumor-centered-patch loader. Sampled validation still predicted zero tumor pixels (`0/54,161` target pixels); loss reduction alone was not treated as improvement, no longer training was started, and the experimental checkpoint remains separate at `checkpoints\best_model_msd_tumor_margin_diag.pt`.

Full-volume inference on all 19 validation cases also predicted zero tumor voxels: pooled tumor Dice, precision, and recall were all 0 for 1,755,272 tumor target voxels in 17 tumor-positive cases. The all-case macro tumor Dice of 0.1053 is only due to the two empty-reference cases receiving an empty/empty score of 1; the positive-reference-case mean and median tumor Dice are both 0. Liver macro Dice was 0.6239, and `liver_74` liver Dice fell to 0.5877 from 0.8047 with the extended checkpoint. The margin experiment did not improve segmentation and is not promoted or selected by the app.

The full-volume results are in `outputs\msd_tumor_margin_diag_val19_metrics_by_patient.csv` and `outputs\msd_tumor_margin_diag_val19_metrics_aggregate.json`, with a visual preview at `outputs\msd_tumor_margin_diag_val19_preview.png`. The evaluator uses four-slice batches and writes per-case tumor/liver Dice, precision, recall, true-positive, predicted-voxel, and target-voxel counts. In cases with tumor targets but no predicted voxels, precision and recall are reported as zero; empty-reference cases are flagged separately.

## Evaluate the model

The external `imagesTs` cases have no labels in this project split and must not be used for development. Evaluate development behavior on the validation partition only:

```powershell
.\.venv\Scripts\python -m src.evaluate --splits data\msd_task03_liver_splits.json --partition val --checkpoint checkpoints\best_model_msd.pt --out-dir outputs
```

This writes per-patient metrics to `outputs\validation_metrics_by_patient.csv` and aggregate metrics to `outputs\validation_metrics_aggregate.json`. It explicitly refuses to use `imagesTs` for Task03 evaluation. For a CPU laptop, full validation inference over every slice can take hours; the training status reports the sampled validation loss instead.

## Launch the Streamlit app

```powershell
.\.venv\Scripts\streamlit run app.py --server.headless true --server.port 8501
```

Then open:

```text
http://localhost:8501
```

The app checks for `checkpoints\best_model_msd.pt` first and will retain the legacy checkpoint as a compatibility fallback.

## Notes and limitations

- The U-Net is 2D and predicts three classes: background, liver, and tumor on axial CT slices.
- The legacy `checkpoints\best_model.pt` and older dataset remain preserved.
- The documented one-epoch command is a resource-conscious workflow check, not evidence of clinical performance or a substitute for a properly trained and evaluated model.
- This is a research prototype and is not a clinically validated medical device.
