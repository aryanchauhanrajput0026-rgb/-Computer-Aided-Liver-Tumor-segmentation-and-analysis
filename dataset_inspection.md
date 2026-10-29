# Selected liver-tumor subset inspection

- Dataset path: `data\Task03_Liver`
- Archive validation: `{"expected_md5": "78a7325e437bda4aacc06eb9b314000e", "status": "not verified in this inspection run"}`
- Files: 8
- Total file size: 460363 bytes
- File formats: `{".nii.gz": 8}`
- NIfTI volumes: 8
- NIfTI image-role files: 4
- NIfTI mask-role files: 4
- 2D image files inventoried: 0
- Structurally valid NIfTI image-mask pairs: 0
- Observed mask label values: `[0, 1, 2]`
- Liver/tumor semantic label mapping: **not verified**

## Dataset documentation
## Per-pair metadata

## Individual NIfTI file metadata
- `data\Task03_Liver\imagesTr\volume-000.nii.gz` (role `image`, candidate `000`): shape `[64, 64, 8]`, spacing `[1.0, 1.0, 1.0] unknown`, orientation `['R', 'A', 'S']`, data type `float32`, label values `None`
- `data\Task03_Liver\imagesTr\volume-001.nii.gz` (role `image`, candidate `001`): shape `[64, 64, 8]`, spacing `[1.0, 1.0, 1.0] unknown`, orientation `['R', 'A', 'S']`, data type `float32`, label values `None`
- `data\Task03_Liver\imagesTr\volume-002.nii.gz` (role `image`, candidate `002`): shape `[64, 64, 8]`, spacing `[1.0, 1.0, 1.0] unknown`, orientation `['R', 'A', 'S']`, data type `float32`, label values `None`
- `data\Task03_Liver\imagesTr\volume-003.nii.gz` (role `image`, candidate `003`): shape `[64, 64, 8]`, spacing `[1.0, 1.0, 1.0] unknown`, orientation `['R', 'A', 'S']`, data type `float32`, label values `None`
- `data\Task03_Liver\labelsTr\volume-000.nii.gz` (role `label`, candidate `volume-000`): shape `[64, 64, 8]`, spacing `[1.0, 1.0, 1.0] unknown`, orientation `['R', 'A', 'S']`, data type `uint8`, label values `[0, 1, 2]`
- `data\Task03_Liver\labelsTr\volume-001.nii.gz` (role `label`, candidate `volume-001`): shape `[64, 64, 8]`, spacing `[1.0, 1.0, 1.0] unknown`, orientation `['R', 'A', 'S']`, data type `uint8`, label values `[0, 1, 2]`
- `data\Task03_Liver\labelsTr\volume-002.nii.gz` (role `label`, candidate `volume-002`): shape `[64, 64, 8]`, spacing `[1.0, 1.0, 1.0] unknown`, orientation `['R', 'A', 'S']`, data type `uint8`, label values `[0, 1, 2]`
- `data\Task03_Liver\labelsTr\volume-003.nii.gz` (role `label`, candidate `volume-003`): shape `[64, 64, 8]`, spacing `[1.0, 1.0, 1.0] unknown`, orientation `['R', 'A', 'S']`, data type `uint8`, label values `[0, 1, 2]`

## Pairing and integrity findings
- Pairing evidence: {"matching_case_ids": [], "exact_shape_and_affine_candidates": [{"image": "data\\Task03_Liver\\imagesTr\\volume-000.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-000.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-000.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-001.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-000.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-002.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-000.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-003.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-001.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-000.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-001.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-001.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-001.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-002.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-001.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-003.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-002.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-000.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-002.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-001.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-002.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-002.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-002.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-003.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-003.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-000.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-003.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-001.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-003.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-002.nii.gz"}, {"image": "data\\Task03_Liver\\imagesTr\\volume-003.nii.gz", "mask": "data\\Task03_Liver\\labelsTr\\volume-003.nii.gz"}]}
- Unmatched images: ["data\\Task03_Liver\\imagesTr\\volume-000.nii.gz", "data\\Task03_Liver\\imagesTr\\volume-001.nii.gz", "data\\Task03_Liver\\imagesTr\\volume-002.nii.gz", "data\\Task03_Liver\\imagesTr\\volume-003.nii.gz"]
- Unmatched labels: ["data\\Task03_Liver\\labelsTr\\volume-000.nii.gz", "data\\Task03_Liver\\labelsTr\\volume-001.nii.gz", "data\\Task03_Liver\\labelsTr\\volume-002.nii.gz", "data\\Task03_Liver\\labelsTr\\volume-003.nii.gz"]
- Duplicate case IDs: []
- Corrupted or invalid pairs: []
- Unclassified NIfTI files: []

## Training blockers
- No structurally valid paired 3D NIfTI volumes found.
- Resolve unmatched, duplicate, or invalid image-mask pairs before splitting.
