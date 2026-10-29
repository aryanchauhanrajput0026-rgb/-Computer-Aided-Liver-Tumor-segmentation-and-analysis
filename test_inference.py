import nibabel as nib
import numpy as np
import torch

from src import inference
from src.evaluate import predict_slice, predict_volume as evaluate_predict_volume
from src.data import normalize_ct


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, images):
        self.batch_sizes.append(images.shape[0])
        logits = torch.zeros(
            (images.shape[0], 3, *images.shape[-2:]),
            device=images.device,
        )
        logits[:, 1] = 1
        return logits


class DummyTumorModel(torch.nn.Module):
    def forward(self, images):
        logits = torch.zeros(
            (images.shape[0], 3, *images.shape[-2:]),
            device=images.device,
        )
        logits[:, 2, 8:10, 11:14] = 5
        return logits


def test_predict_volume_batches_slices_and_reports_progress(tmp_path, monkeypatch):
    scan_path = tmp_path / "scan.nii"
    scan = nib.Nifti1Image(np.zeros((32, 32, 5), dtype=np.float32), np.eye(4))
    nib.save(scan, scan_path)
    model = DummyModel()
    monkeypatch.setattr(
        inference,
        "load_model",
        lambda checkpoint: (model, "cpu", {"preprocessing": {"hu_min": -100, "hu_max": 400}}),
    )
    progress = []

    _, prediction, _ = inference.predict_volume(
        scan_path,
        "unused-checkpoint",
        progress_callback=lambda completed, total: progress.append((completed, total)),
        batch_size=2,
    )

    assert model.batch_sizes == [2, 2, 1]
    assert progress == [(2, 5), (4, 5), (5, 5)]
    assert np.all(prediction == 1)


def test_validation_volume_batching_matches_per_slice_inference():
    model = DummyModel()
    volume = np.arange(32 * 32 * 5, dtype=np.float32).reshape(32, 32, 5)
    preprocessing = {"hu_min": -100, "hu_max": 400}

    expected = np.stack(
        [
            predict_slice(
                model,
                normalize_ct(volume[:, :, z], **preprocessing),
                "cpu",
            )
            for z in range(volume.shape[2])
        ],
        axis=2,
    )
    model.batch_sizes.clear()

    prediction = evaluate_predict_volume(
        model, volume, "cpu", preprocessing, batch_size=2
    )

    assert model.batch_sizes == [2, 2, 1]
    assert np.array_equal(prediction, expected)


def test_tumor_class_survives_inference_reconstruction_measurements_and_save(
    tmp_path, monkeypatch
):
    scan_path = tmp_path / "scan.nii.gz"
    scan = nib.Nifti1Image(np.zeros((32, 32, 3), dtype=np.float32), np.eye(4))
    scan.header.set_xyzt_units("mm")
    nib.save(scan, scan_path)
    model = DummyTumorModel()
    monkeypatch.setattr(
        inference,
        "load_model",
        lambda checkpoint: (
            model,
            "cpu",
            {"preprocessing": {"hu_min": -100, "hu_max": 400}},
        ),
    )

    _, prediction, reference = inference.predict_volume(
        scan_path,
        "unused-checkpoint",
        batch_size=2,
    )
    metrics = inference.measurements(prediction, reference)
    output_path = tmp_path / "prediction.nii.gz"
    inference.save_mask(prediction, reference, output_path)
    saved = nib.load(output_path)
    saved_prediction = np.asarray(saved.dataobj)

    assert prediction.shape == (32, 32, 3)
    assert np.unique(prediction).tolist() == [0, 2]
    assert int((prediction == 2).sum()) == 18
    assert metrics["tumor_voxels"] == 18
    assert np.array_equal(saved_prediction, prediction)
    assert np.allclose(saved.affine, reference.affine)
