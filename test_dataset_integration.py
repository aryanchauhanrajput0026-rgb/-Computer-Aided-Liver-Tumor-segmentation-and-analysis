import json
import sys
import zipfile

import nibabel as nib
import numpy as np
import pytest
import torch

from src.data import CaseBatchSampler, SliceDataset, validate_label
from src.train import (
    class_dice_metrics,
    dice_loss,
    tumor_margin_loss,
    update_class_dice_counts,
)
from src.inspect_dataset import (
    EXPECTED_MD5,
    _safe_extract,
    inspect_dataset,
    main as inspect_main,
    md5_file,
)
from src.make_splits import create_msd_splits, create_splits


def _write_case(root, case_id, raw_labels=(0, 1, 2), image_shape=(17, 19, 3)):
    image_dir = root / "imagesTr"
    label_dir = root / "labelsTr"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image = np.arange(np.prod(image_shape), dtype=np.float32).reshape(image_shape)
    label = np.zeros(image_shape, dtype=np.uint8)
    label[2:5, 3:6, :] = raw_labels[1]
    label[7:9, 8:10, :] = raw_labels[2]
    affine = np.diag([0.7, 0.8, 2.5, 1.0])
    image_path = image_dir / f"{case_id}.nii.gz"
    label_path = label_dir / f"{case_id}.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(label, affine), label_path)
    return image_path, label_path


def test_inspection_reports_pairing_metadata_and_labels(tmp_path):
    root = tmp_path / "liver-tumor-subset"
    _write_case(root, "case-001")

    report = inspect_dataset(root)

    assert report["valid_nifti_pair_count"] == 1
    assert report["nifti_image_count"] == report["nifti_mask_count"] == 1
    assert report["observed_mask_label_values"] == [0, 1, 2]
    assert report["pairs"][0]["image_metadata"]["shape"] == [17, 19, 3]
    assert np.allclose(
        report["pairs"][0]["image_metadata"]["spacing"], [0.7, 0.8, 2.5]
    )
    assert report["label_validation"]["training_ready"] is False
    assert not report["training_blockers"]


def test_split_is_deterministic_and_keeps_cases_disjoint(tmp_path):
    root = tmp_path / "liver-tumor-subset"
    for index in range(5):
        _write_case(root, f"case-{index:03}")
    report = inspect_dataset(root)

    split_a = create_splits(report, 4, 42, 1, 2)
    split_b = create_splits(report, 4, 42, 1, 2)

    assert split_a == split_b
    assert [len(split_a[key]) for key in ("train", "val", "test")] == [2, 1, 1]
    ids = [item["id"] for key in ("train", "val", "test") for item in split_a[key]]
    assert len(ids) == len(set(ids))
    assert split_a["raw_label_map"] == {"0": 0, "1": 1, "2": 2}


def test_slice_loader_maps_raw_values_and_reads_slices(tmp_path):
    root = tmp_path / "liver-tumor-subset"
    image_path, label_path = _write_case(root, "case-001", raw_labels=(0, 4, 7))
    dataset = SliceDataset(
        [{
            "id": "case-001",
            "image": str(image_path),
            "label": str(label_path),
            "raw_label_map": {"0": 0, "4": 1, "7": 2},
        }]
    )

    image, label = dataset[0]

    assert len(dataset) == 3
    assert tuple(image.shape) == (1, 32, 32)
    assert tuple(label.shape) == (32, 32)
    assert set(np.unique(label.numpy())) == {0, 1, 2}


def test_case_batch_sampler_keeps_batches_within_one_patient(tmp_path):
    root = tmp_path / "liver-tumor-subset"
    first = _write_case(root, "case-001")
    second = _write_case(root, "case-002")
    pairs = [
        {"id": case_id, "image": str(image), "label": str(label)}
        for case_id, (image, label) in zip(("case-001", "case-002"), (first, second))
    ]
    dataset = SliceDataset(pairs)
    sampler = CaseBatchSampler(dataset, batch_size=2, shuffle=True, seed=17)

    batches = list(sampler)

    assert len(batches) == len(sampler)
    assert all(
        len({dataset.index[index][0]["id"] for index in batch}) == 1
        for batch in batches
    )
    assert sorted(index for batch in batches for index in batch) == list(range(len(dataset)))


def test_slice_loader_uses_tumor_centered_training_patches(tmp_path):
    root = tmp_path / "liver-tumor-subset"
    image_path, label_path = _write_case(root, "case-001")
    dataset = SliceDataset(
        [{
            "id": "case-001",
            "image": str(image_path),
            "label": str(label_path),
        }],
        augment=True,
        patch_size=16,
    )

    image, label = dataset[0]

    assert tuple(image.shape) == (1, 16, 16)
    assert tuple(label.shape) == (16, 16)
    assert 2 in np.unique(label.numpy())


def test_slice_sampler_prioritizes_tumor_slices_and_keeps_context(tmp_path):
    root = tmp_path / "liver-tumor-subset"
    image_path, label_path = _write_case(root, "case-001", image_shape=(17, 19, 8))
    mask = np.zeros((17, 19, 8), dtype=np.uint8)
    mask[7:9, 8:10, 1:7] = 2
    nib.save(nib.Nifti1Image(mask, np.diag([0.7, 0.8, 2.5, 1.0])), label_path)
    dataset = SliceDataset([{
        "id": "case-001",
        "image": str(image_path),
        "label": str(label_path),
    }])

    sampled = dataset.sample_case_indices(dataset.case_indices["case-001"], 4)
    sampled_z = [dataset.index[index][1] for index in sampled]

    assert sum(np.any(mask[:, :, z] == 2) for z in sampled_z) == 3
    assert sum(not np.any(mask[:, :, z] == 2) for z in sampled_z) == 1


def test_class_dice_metrics_measure_liver_and_tumor_logits():
    target = torch.tensor([[[0, 1], [2, 2]]])
    logits = torch.full((1, 3, 2, 2), -5.0)
    logits[:, 0, 0, 0] = 5.0
    logits[:, 1, 0, 1] = 5.0
    logits[:, 2, 1, 0] = 5.0
    logits[:, 1, 1, 1] = 5.0
    counts = {1: [0, 0, 0], 2: [0, 0, 0]}

    update_class_dice_counts(counts, logits, target)
    metrics = class_dice_metrics(counts)

    assert metrics["liver_dice"] == pytest.approx(2 / 3)
    assert metrics["tumor_dice"] == pytest.approx(2 / 3)
    assert metrics["tumor_predicted_pixels"] == 1
    assert metrics["tumor_target_pixels"] == 2


def test_dice_loss_does_not_penalize_classes_absent_from_target():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    target = torch.zeros((1, 4, 4), dtype=torch.long)

    loss = dice_loss(logits, target)
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert torch.count_nonzero(logits.grad).item() == 0


def test_tumor_margin_loss_directly_pushes_tumor_above_liver():
    logits = torch.tensor([[[[0.0]], [[2.0]], [[0.0]]]], requires_grad=True)
    target = torch.tensor([[[2]]])

    loss = tumor_margin_loss(logits, target, margin=1.0)
    loss.backward()

    assert loss.item() == pytest.approx(3.0)
    assert logits.grad[0, 2, 0, 0].item() == pytest.approx(-1.0)
    assert logits.grad[0, 1, 0, 0].item() == pytest.approx(1.0)
    assert logits.grad[0, 0, 0, 0].item() == pytest.approx(0.0)


def test_tumor_margin_loss_is_zero_without_tumor_targets():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    target = torch.zeros((1, 4, 4), dtype=torch.long)

    loss = tumor_margin_loss(logits, target)
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert torch.count_nonzero(logits.grad).item() == 0


def test_label_validation_rejects_unmapped_and_non_integer_values():
    with pytest.raises(ValueError, match="not present in the verified mapping"):
        validate_label(np.array([[0, 5]]), {"0": 0, "1": 1, "2": 2})
    with pytest.raises(ValueError, match="non-finite or non-integer"):
        validate_label(np.array([[0, 1.5]]))


def test_inspection_reports_unmatched_or_misaligned_pairs(tmp_path):
    root = tmp_path / "liver-tumor-subset"
    _, label_path = _write_case(root, "case-001")
    nib.save(
        nib.Nifti1Image(
            np.zeros((17, 19, 3), dtype=np.uint8),
            np.diag([1.0, 1.0, 1.0, 1.0]),
        ),
        label_path,
    )

    report = inspect_dataset(root)

    assert report["valid_nifti_pair_count"] == 0
    assert report["corrupted_or_invalid_pairs"][0]["error"] == (
        "Image and mask dimensions or affine differ"
    )
    assert report["training_blockers"]


def test_split_generation_stops_on_unmatched_dataset_report(tmp_path):
    root = tmp_path / "liver-tumor-subset"
    _write_case(root, "case-001")
    report = inspect_dataset(root)
    report["training_blockers"] = ["No structurally valid paired volumes"]

    with pytest.raises(ValueError, match="training blockers"):
        create_splits(report, 3, 42, 1, 2)


def test_msd_task03_liver_manifest_is_valid(tmp_path):
    root = tmp_path / "Task03_Liver"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir(parents=True)
    (root / "imagesTs").mkdir(parents=True)
    dataset = {
        "name": "Liver",
        "labels": {"0": "background", "1": "liver", "2": "cancer"},
        "numTraining": 1,
        "numTest": 1,
        "training": [
            {"image": "./imagesTr/liver_0.nii.gz", "label": "./labelsTr/liver_0.nii.gz"}
        ],
    }
    (root / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")

    image = np.zeros((10, 10, 5), dtype=np.float32)
    affine = np.eye(4)
    mask = np.zeros((10, 10, 5), dtype=np.uint8)
    mask[2:5, 2:5, :] = 1
    mask[6:8, 6:8, :] = 2
    nib.save(nib.Nifti1Image(image, affine), root / "imagesTr" / "liver_0.nii.gz")
    nib.save(nib.Nifti1Image(mask, affine), root / "labelsTr" / "liver_0.nii.gz")
    nib.save(nib.Nifti1Image(image, affine), root / "imagesTs" / "liver_test_0.nii.gz")

    report = inspect_dataset(root)

    assert report["valid_nifti_pair_count"] == 1
    assert report["observed_mask_label_values"] == [0, 1, 2]
    assert report["pairs"][0]["id"] == "liver_0"
    assert not report["training_blockers"]


def test_msd_split_uses_manifest_and_keeps_external_test_separate(tmp_path):
    root = tmp_path / "Task03_Liver"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir(parents=True)
    (root / "imagesTs").mkdir(parents=True)
    training = []
    for index in range(3):
        image_path, label_path = _write_case(root, f"liver_{index}")
        training.append({
            "image": f"./imagesTr/{image_path.name}",
            "label": f"./labelsTr/{label_path.name}",
        })
    test_paths = []
    for index in range(2):
        test_path = root / "imagesTs" / f"liver_test_{index}.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((17, 19, 3), dtype=np.float32), np.eye(4)),
            test_path,
        )
        test_paths.append(test_path)
    (root / "imagesTr" / "._metadata.nii.gz").write_bytes(b"ignored metadata")
    (root / "dataset.json").write_text(
        json.dumps({
            "name": "Liver",
            "labels": {"0": "background", "1": "liver", "2": "cancer"},
            "numTraining": 3,
            "numTest": 2,
            "training": training,
        }),
        encoding="utf-8",
    )

    split = create_msd_splits(root, seed=42)
    repeated = create_msd_splits(root, seed=42)

    assert split == repeated
    assert [len(split[key]) for key in ("train", "val", "test")] == [2, 1, 0]
    assert split["external_test_case_count"] == 2
    assert {item["image"] for item in split["external_test_images"]} == {
        str(path) for path in test_paths
    }
    train_ids = {item["id"] for item in split["train"]}
    val_ids = {item["id"] for item in split["val"]}
    external_test_ids = {item["id"] for item in split["external_test_images"]}
    assert not train_ids & val_ids
    assert not train_ids & external_test_ids
    assert not val_ids & external_test_ids


def test_msd_split_rejects_misaligned_image_mask(tmp_path):
    root = tmp_path / "Task03_Liver"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir(parents=True)
    (root / "imagesTs").mkdir(parents=True)
    training = []
    for index in range(3):
        image_path, label_path = _write_case(root, f"liver_{index}")
        training.append({
            "image": f"./imagesTr/{image_path.name}",
            "label": f"./labelsTr/{label_path.name}",
        })
    nib.save(
        nib.Nifti1Image(np.zeros((17, 19, 3), dtype=np.uint8), np.diag([2, 1, 1, 1])),
        root / "labelsTr" / "liver_1.nii.gz",
    )
    (root / "dataset.json").write_text(
        json.dumps({
            "name": "Liver",
            "labels": {"0": "background", "1": "liver", "2": "cancer"},
            "numTraining": 3,
            "numTest": 0,
            "training": training,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="affine mismatch"):
        create_msd_splits(root)


def test_checksum_helper_returns_expected_digest_for_known_content(tmp_path):
    archive = tmp_path / "selected.zip"
    archive.write_bytes(b"dataset archive bytes")

    assert md5_file(archive) != EXPECTED_MD5


def test_checksum_mismatch_stops_before_extraction(tmp_path, monkeypatch):
    archive = tmp_path / "wrong.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("images/case.nii.gz", b"not a NIfTI")
    destination = tmp_path / "data" / "liver-tumor-subset"
    monkeypatch.setattr(
        sys,
        "argv",
        ["inspect_dataset", "--archive", str(archive), "--root", str(destination)],
    )

    with pytest.raises(SystemExit, match="MD5 mismatch"):
        inspect_main()

    assert not destination.exists()


def test_safe_extraction_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "unsafe")
    destination = tmp_path / "extracted"

    with pytest.raises(ValueError, match="Unsafe path"):
        _safe_extract(archive, destination)

    assert not (tmp_path / "outside.txt").exists()
