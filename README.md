# MGR Attention + RNO

Official implementation of **Improving Diffusion-Based Augmentation for Data-Scarce Image Classification via Mask-Guided Reference Attention and Repulsive Noise Optimization**<br>
Hyun-Jic Oh\*, GaYeon Koh\*, Won-Ki Jeong<br>
*IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2027* (Round 1 accept)<br>
(\* equal contribution)

[![Project Page](https://img.shields.io/badge/Project_Page-MGRAttn--RNO-1f5fa9?style=for-the-badge&logo=githubpages&logoColor=white)](https://hvcl.github.io/MGRAttn-RNO/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.04242-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.04242)

This repository provides the sampling pipeline for diffusion-based data augmentation with two sampling-time modules:

- **Mask-Guided Reference (MGR) Attention** updates only the foreground value tokens of the target image with reference features, using zero-shot segmentation (SAM3) to localize the foreground.
- **Repulsive Noise Optimization (RNO)** optimizes the initial noises before denoising with a repulsion-only objective against adaptation-prompt anchors, encouraging diverse prompt-conditioned layouts.

Both modules operate at sampling time on top of an adapted (concept-learned) Stable Diffusion v1.5 model and do not modify any model weights.

The release includes ready-to-run examples for CUB-LT IF100 (`imb_cub_0.01`) and Flower-LT IF100 (`imb_flower_0.01`). Sampling runs in three stages:

1. Dump intermediate target images and latents at the MGR start step (RNO is applied before this stage's denoising).
2. Extract SAM3 target masks from the dumped intermediate images.
3. Resume sampling from the dumped latents with MGR Attention using the extracted target masks.

> **Note on timestep notation.** `STEP_START_MGR=20` is the denoising-step index in 50-step DDIM sampling. It corresponds to the mask extraction timestep $t_m=30$ in the paper.

---

## 1. Repository Structure

```text
MGRAttn-RNO/
├── assets/
│   ├── ft_ti_db/<dataset>/            # download from Diff-II (see Section 2.1)
│   ├── ref_masks/<dataset>/<class>/<stem>/{input_image.png, mask_binary.png}
│   ├── suffix/<dataset>/suffix.txt    # download from Diff-II (see Section 2.1)
│   └── final_ref_mask_manifest.csv
├── envs/
│   ├── environment_mgr.yml
│   └── environment_sam3.yml
├── scripts/
│   ├── 01_dump_mgr_rno.sh
│   ├── run_sam3.sh
│   ├── 02_resume_mgr_rno.sh
│   └── setup_sam3_assets.sh
└── src/
    ├── sampling/
    │   ├── run_mgr_rno_sampling.py
    │   ├── utils_sd15.py
    │   └── pipelines/
    │       ├── pipeline_sd15_MGR_RNO.py
    │       └── inverted_ve_pipeline_MGR_RNO.py
    └── segmentation/
        └── extract_sam3_policy_masks.py
```

Outputs are written to `outputs/` at runtime.

---

## 2. Assets and Manifest Format

The sampler uses `assets/final_ref_mask_manifest.csv` to locate the reference image and reference foreground mask for each sample.

Expected CSV columns:

```text
dataset,class,stem,status,ref_image_path,ref_mask_path
```

Example row:

```csv
imb_cub_0.01,Blue_winged_Warbler,1198,ready,assets/ref_masks/imb_cub_0.01/Blue_winged_Warbler/1198/input_image.png,assets/ref_masks/imb_cub_0.01/Blue_winged_Warbler/1198/mask_binary.png
```

The path columns should be repository-relative paths. Absolute paths in the CSV are rejected by the sampler.

The fine-tuned checkpoints under `assets/ft_ti_db/` and the prompt suffixes under `assets/suffix/` are not included in this repository. Download them from [Diff-II](https://github.com/HKUST-LongGroup/Diff-II) as described below.

### 2.1 Download Diff-II checkpoints and suffixes

```bash
wget https://huggingface.co/datasets/Luka-Wang/Diff-II/resolve/main/ft_ti_db.zip
wget https://huggingface.co/datasets/Luka-Wang/Diff-II/resolve/main/suffix.zip
unzip ft_ti_db.zip -d diffii_tmp && unzip suffix.zip -d diffii_tmp
```

Place the files for each dataset so that the following paths exist:

```text
assets/ft_ti_db/imb_cub_0.01/pytorch_lora_weights.safetensors
assets/ft_ti_db/imb_cub_0.01/learned_embeds-steps-35000.bin
assets/ft_ti_db/imb_flower_0.01/pytorch_lora_weights.safetensors
assets/ft_ti_db/imb_flower_0.01/learned_embeds-steps-35000.bin
assets/suffix/imb_cub_0.01/suffix.txt
assets/suffix/imb_flower_0.01/suffix.txt
```

To keep the repository small, `assets/ref_masks/` ships only the reference image/mask pairs for the default demonstration classes (Section 3). The directory layout is preserved, and `assets/final_ref_mask_manifest.csv` indexes the shipped pairs.

The `dataset` field must match the dataset key used by the scripts:

| Dataset | Script arguments | Manifest `dataset` value |
|---|---|---|
| CUB-LT IF100 | `cub 0.01` | `imb_cub_0.01` |
| Flower-LT IF100 | `flower 0.01` | `imb_flower_0.01` |

---

## 3. Dataset-Specific Defaults

The scripts automatically select the following dataset-specific MGR/RNO hyperparameters.

| Dataset | `MGR_BLOCK_FROM` | `RNO_LR` | `RNO_STEPS` | `RNO_REPEL_WEIGHT` | Default demo classes |
|---|---:|---:|---:|---:|---|
| `imb_cub_0.01` | `up2_2` | `0.01` | `3` | `1.0` | `Blue_winged_Warbler,Common_Yellowthroat,Le_Conte_Sparrow` |
| `imb_flower_0.01` | `up2_1` | `0.08` | `30` | `2.0` | `toad_lily,bearded_iris,blanket_flower` |

Common defaults:

```text
STEP_START_MGR=20
GENERATION_MODE=normalization
TARGET_BATCH_SIZE=4
GUIDANCE_SCALE=7.0
NUM_INFERENCE_STEPS=50
RNO_GAMMA=3.0
RNO_POSITIVE_WEIGHT=0.0
USE_DEMO_QUOTA=1
DEMO_QUOTA_PER_CLASS=50
```

The RNO hyperparameters are used by Stage 1, and they are also kept in `run_sam3.sh` and `02_resume_mgr_rno.sh` to reconstruct the same `METHOD_TAG` and locate the matching dump/mask folders. The resume stage itself does not pass `--use_rno`.

For the lightweight demo, the scripts use selected tail classes with an explicit per-class quota while keeping the generation mode as `normalization`.

This release ships reference image/mask pairs and manifest rows for the default demonstration classes only. Running with `ONLY_CLASSES=all` will therefore sample only those demonstration classes. Full long-tailed reproduction over all classes requires regenerating `assets/final_ref_mask_manifest.csv` and the corresponding `assets/ref_masks/` reference pairs for the remaining classes:

```bash
ONLY_CLASSES=all USE_DEMO_QUOTA=0 \
bash scripts/01_dump_mgr_rno.sh cub 0.01 0 0 44
```

---

## 4. Environment Setup

The pipeline uses two environments:

- `mgr_env`: Stable Diffusion, MGR, and RNO sampling.
- `sam3_env`: SAM3-based target-mask extraction.

### 4.1 MGR/RNO sampling environment

```bash
conda env create -f envs/environment_mgr.yml
conda activate mgr_env

export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH
```

Check installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import diffusers, transformers; print(diffusers.__version__, transformers.__version__)"
```

### 4.2 SAM3 segmentation environment

```bash
conda env create -f envs/environment_sam3.yml
conda activate sam3_env

export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH
bash scripts/setup_sam3_assets.sh
```

`setup_sam3_assets.sh` places the BPE vocabulary file required by the installed SAM3 package at the expected location. If the file is already included at `assets/sam3/bpe_simple_vocab_16e6.txt.gz`, the script copies it locally. Otherwise, it attempts to download the public CLIP BPE vocabulary file.

Check installation:

```bash
python -c "import numpy, cv2, torch, sam3; print(numpy.__version__, cv2.__version__, torch.__version__, torch.cuda.is_available())"
```

---

## 5. Pipeline Overview

The full generation pipeline consists of three stages.

### Stage 1: Dump intermediate target images and latents

This stage runs denoising up to `STEP_START_MGR`, applies RNO before the dump, and saves the intermediate target images/latents.

```bash
conda activate mgr_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH

bash scripts/01_dump_mgr_rno.sh <DATA> <IMB_DATA_RATIO> <PROC_ID> <GPU_ID> <SEED>
```

### Stage 2: Extract SAM3 target masks

This stage runs SAM3 on the dumped intermediate target images.

```bash
conda activate sam3_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH

bash scripts/run_sam3.sh <DATA> <IMB_DATA_RATIO> <GPU_ID> <SEED>
```

### Stage 3: Resume sampling with target masks

This stage resumes from the dumped latents and uses the SAM3 target masks. RNO is intentionally not enabled in this stage because it has already been applied before the Stage-1 dump. The RNO hyperparameters are still used to reconstruct the Stage-1 `METHOD_TAG` and locate the correct dump and mask directories.

```bash
conda activate mgr_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH

bash scripts/02_resume_mgr_rno.sh <DATA> <IMB_DATA_RATIO> <PROC_ID> <GPU_ID> <SEED>
```

---

## 6. Running Examples

### 6.1 CUB-LT IF100 (`imb_cub_0.01`)

```bash
# Stage 1: dump step-20 targets
conda activate mgr_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH
bash scripts/01_dump_mgr_rno.sh cub 0.01 0 0 44

# Stage 2: SAM3 target-mask extraction
conda activate sam3_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH
bash scripts/run_sam3.sh cub 0.01 0 44

# Stage 3: resume sampling
conda activate mgr_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH
bash scripts/02_resume_mgr_rno.sh cub 0.01 0 0 44
```

Expected output directories:

```text
outputs/imb_cub_0.01/
├── MGR_RNO_normalization_seed44_lr0.01-optsteps3-rnogamma3.0-repelweight1.0-pos0.0_step20_dump/
├── MGR_RNO_normalization_seed44_lr0.01-optsteps3-rnogamma3.0-repelweight1.0-pos0.0_step20_dump_sam3_masks/
└── MGR_RNO_normalization_seed44_lr0.01-optsteps3-rnogamma3.0-repelweight1.0-pos0.0/
```

### 6.2 Flower-LT IF100 (`imb_flower_0.01`)

```bash
# Stage 1: dump step-20 targets
conda activate mgr_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH
bash scripts/01_dump_mgr_rno.sh flower 0.01 0 0 44

# Stage 2: SAM3 target-mask extraction
conda activate sam3_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH
bash scripts/run_sam3.sh flower 0.01 0 44

# Stage 3: resume sampling
conda activate mgr_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH
bash scripts/02_resume_mgr_rno.sh flower 0.01 0 0 44
```

Expected output directories:

```text
outputs/imb_flower_0.01/
├── MGR_RNO_normalization_seed44_lr0.08-optsteps30-rnogamma3.0-repelweight2.0-pos0.0_step20_dump/
├── MGR_RNO_normalization_seed44_lr0.08-optsteps30-rnogamma3.0-repelweight2.0-pos0.0_step20_dump_sam3_masks/
└── MGR_RNO_normalization_seed44_lr0.08-optsteps30-rnogamma3.0-repelweight2.0-pos0.0/
```

---

## 7. Smoke Test

For a quick functionality check, reduce the per-class demo quota and target batch size.

### CUB-LT smoke test

```bash
conda activate mgr_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH

DEMO_QUOTA_PER_CLASS=4 \
TARGET_BATCH_SIZE=1 \
bash scripts/01_dump_mgr_rno.sh cub 0.01 0 0 44
```

Then extract SAM3 masks:

```bash
conda activate sam3_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH

bash scripts/run_sam3.sh cub 0.01 0 44
```

Finally resume sampling:

```bash
conda activate mgr_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH

DEMO_QUOTA_PER_CLASS=4 \
TARGET_BATCH_SIZE=1 \
bash scripts/02_resume_mgr_rno.sh cub 0.01 0 0 44
```

### Flower-LT smoke test

```bash
conda activate mgr_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH

DEMO_QUOTA_PER_CLASS=4 \
TARGET_BATCH_SIZE=1 \
bash scripts/01_dump_mgr_rno.sh flower 0.01 0 0 44
```

Then extract SAM3 masks:

```bash
conda activate sam3_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH

bash scripts/run_sam3.sh flower 0.01 0 44
```

Finally resume sampling:

```bash
conda activate mgr_env
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH

DEMO_QUOTA_PER_CLASS=4 \
TARGET_BATCH_SIZE=1 \
bash scripts/02_resume_mgr_rno.sh flower 0.01 0 0 44
```

---

## 8. Script Arguments

### `01_dump_mgr_rno.sh`

```bash
bash scripts/01_dump_mgr_rno.sh <DATA> <IMB_DATA_RATIO> <PROC_ID> <GPU_ID> <SEED>
```

Examples:

```bash
bash scripts/01_dump_mgr_rno.sh cub 0.01 0 0 44
bash scripts/01_dump_mgr_rno.sh flower 0.01 0 0 44
```

### `run_sam3.sh`

```bash
bash scripts/run_sam3.sh <DATA> <IMB_DATA_RATIO> <GPU_ID> <SEED>
```

Examples:

```bash
bash scripts/run_sam3.sh cub 0.01 0 44
bash scripts/run_sam3.sh flower 0.01 0 44
```

### `02_resume_mgr_rno.sh`

```bash
bash scripts/02_resume_mgr_rno.sh <DATA> <IMB_DATA_RATIO> <PROC_ID> <GPU_ID> <SEED>
```

Examples:

```bash
bash scripts/02_resume_mgr_rno.sh cub 0.01 0 0 44
bash scripts/02_resume_mgr_rno.sh flower 0.01 0 0 44
```

---

## 9. Common Overrides

The following environment variables can be used with the scripts.

```bash
ONLY_CLASSES="Blue_winged_Warbler"
MAX_CLASSES=1
TARGET_BATCH_SIZE=4
STEP_START_MGR=20
GENERATION_MODE=normalization
USE_DEMO_QUOTA=1
DEMO_QUOTA_PER_CLASS=50
RNO_GAMMA=3.0
RNO_POSITIVE_WEIGHT=0.0
N_PROC=1
```

Dataset-specific defaults can also be manually overridden if needed:

```bash
MGR_BLOCK_FROM=up2_2
RNO_LR=0.01
RNO_STEPS=3
RNO_REPEL_WEIGHT=1.0
```

Example:

```bash
ONLY_CLASSES="Blue_winged_Warbler" \
bash scripts/01_dump_mgr_rno.sh cub 0.01 0 0 44
```

For the two long-tailed datasets shown here, the default generation mode is `normalization`.

---

## 10. Multi-GPU / Sharded Execution

The scripts support process-level sharding using `N_PROC` and `PROC_ID`.

Example with two processes:

```bash
# GPU 0
N_PROC=2 bash scripts/01_dump_mgr_rno.sh cub 0.01 0 0 44

# GPU 1
N_PROC=2 bash scripts/01_dump_mgr_rno.sh cub 0.01 1 1 44
```

Run the same sharding pattern for Stage 3.

SAM3 target-mask extraction can be run after all Stage-1 dump folders are available:

```bash
bash scripts/run_sam3.sh cub 0.01 0 44
```

---

## 11. Notes on Reproducibility

- The default seed is `44`.
- The scripts automatically set dataset-specific MGR/RNO defaults for `imb_cub_0.01` and `imb_flower_0.01`.
- The same `DATA`, `IMB_DATA_RATIO`, `SEED`, `GENERATION_MODE`, and RNO hyperparameters must be used across the three stages.
- Stage 2 must point to the dump directory produced by Stage 1.
- Stage 3 must point to both the Stage-1 dump directory and the Stage-2 SAM3 mask directory.
- RNO is enabled in Stage 1 and intentionally disabled in Stage 3. The RNO hyperparameters are still used in Stage 3 only to reconstruct the method tag and locate the Stage-1 dump/mask folders.
- If `data/<dataset>/train` is not available, the sampler uses `assets/final_ref_mask_manifest.csv` as a lightweight train index.

---

## 12. External Dependency: SAM3

SAM3 is used only for target-mask extraction from decoded intermediate images. The MGR/RNO sampling code does not depend on SAM3 after target masks have been precomputed.

This repository does not vendor the SAM3 source code. Install SAM3 using the provided `envs/environment_sam3.yml` or follow the official public SAM3 installation instructions. The script `scripts/setup_sam3_assets.sh` is provided only to place the required tokenizer asset at the location expected by the installed SAM3 package.

---

## 13. Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`

Install OpenCV in the relevant environment:

```bash
pip install opencv-python-headless==4.10.0.84
```

### `sam3` fails with missing `bpe_simple_vocab_16e6.txt.gz`

Run:

```bash
conda activate sam3_env
bash scripts/setup_sam3_assets.sh
```

### `DUMP_ROOT does not exist`

Run Stage 1 first with the same dataset, seed, and dataset-specific hyperparameters:

```bash
bash scripts/01_dump_mgr_rno.sh cub 0.01 0 0 44
```

### `TARGET_MASK_ROOT does not exist`

Run SAM3 target-mask extraction:

```bash
bash scripts/run_sam3.sh cub 0.01 0 44
```

### Sampling is slow

Use the smoke-test setting with a smaller `DEMO_QUOTA_PER_CLASS` and `TARGET_BATCH_SIZE` for a quick functionality check.

---

---

## Citation

```bibtex
@inproceedings{oh2027improving,
  title     = {Improving Diffusion-Based Augmentation for Data-Scarce Image Classification via Mask-Guided Reference Attention and Repulsive Noise Optimization},
  author    = {Oh, Hyun-Jic and Koh, GaYeon and Jeong, Won-Ki},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

---

## License

This repository is released under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE). It may be used for non-commercial research and educational purposes. For commercial use, please contact the authors.

Files adapted from third-party projects remain under their original licenses, listed below.

### Third-Party Code

- `src/sampling/pipelines/pipeline_sd15_MGR_RNO.py` is modified from the Stable Diffusion pipeline in [diffusers](https://github.com/huggingface/diffusers) (Apache License 2.0).
- `src/sampling/pipelines/inverted_ve_pipeline_MGR_RNO.py` and `src/sampling/utils_sd15.py` are adapted from [Visual Style Prompting](https://github.com/naver-ai/Visual-Style-Prompting) (Apache License 2.0).

### Third-Party Assets

The following assets are not produced by this work. Use them according to the original projects' licenses and terms.

- **Stable Diffusion v1.5** (`runwayml/stable-diffusion-v1-5`): base text-to-image model, downloaded at runtime.
- **SAM3**: used only for target-mask extraction. This repository does not vendor SAM3 source code.
- **Fine-tuned checkpoints** `assets/ft_ti_db/<dataset>/` (LoRA weights and textual-inversion embeddings) and **prompt suffixes** `assets/suffix/<dataset>/suffix.txt`: released by [Diff-II](https://github.com/HKUST-LongGroup/Diff-II). Not redistributed here; download them as described in Section 2.1.
- **Reference images** under `assets/ref_masks/` are taken from [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) and [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and are subject to those datasets' terms of use.

## Acknowledgments

We thank the authors of diffusers, Visual Style Prompting, Diff-II, and SAM3 for releasing their code and models.
