#!/usr/bin/env bash
set -euo pipefail

# Extract SAM3 target masks from dumped step-MGR target images.
#
# Supported datasets:
#   - CUB-LT IF100    : DATA=cub,    IMB_DATA_RATIO=0.01 -> DATASET=imb_cub_0.01
#   - Flower-LT IF100 : DATA=flower, IMB_DATA_RATIO=0.01 -> DATASET=imb_flower_0.01
#
# Usage:
#   bash scripts/run_sam3.sh cub 0.01 0 44
#   bash scripts/run_sam3.sh flower 0.01 0 44

PYTHON=${PYTHON:-python}

DATA=${1:?"DATA is required, e.g., cub or flower"}
IMB_DATA_RATIO=${2:?"IMB_DATA_RATIO is required, e.g., 0.01"}
GPU_ID=${3:-0}
SEED=${4:-44}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SCRIPT="${PROJECT_ROOT}/src/segmentation/extract_sam3_policy_masks.py"
POLICY_CSV="${POLICY_CSV:-${PROJECT_ROOT}/assets/final_ref_mask_manifest.csv}"

DATASET="imb_${DATA}_${IMB_DATA_RATIO}"

# ------------------------------------------------------------
# Dataset-specific defaults.
# These must match Stage 1/Stage 2 because they define METHOD_TAG.
# ------------------------------------------------------------
case "${DATASET}" in
  "imb_cub_0.01")
    RNO_LR=${RNO_LR:-0.01}
    RNO_STEPS=${RNO_STEPS:-3}
    RNO_REPEL_WEIGHT=${RNO_REPEL_WEIGHT:-1.0}
    SAM3_PROMPT=${PROMPT:-bird}
    ;;
  "imb_flower_0.01")
    RNO_LR=${RNO_LR:-0.08}
    RNO_STEPS=${RNO_STEPS:-30}
    RNO_REPEL_WEIGHT=${RNO_REPEL_WEIGHT:-2.0}
    SAM3_PROMPT=${PROMPT:-flower}
    ;;
  *)
    echo "[ERROR] Unsupported dataset: ${DATASET}" >&2
    echo "Supported datasets are:" >&2
    echo "  - imb_cub_0.01    via: bash scripts/run_sam3.sh cub 0.01 ..." >&2
    echo "  - imb_flower_0.01 via: bash scripts/run_sam3.sh flower 0.01 ..." >&2
    exit 1
    ;;
esac

STEP_START_MGR=${STEP_START_MGR:-20}
GENERATION_MODE=${GENERATION_MODE:-normalization}
RNO_GAMMA=${RNO_GAMMA:-3.0}
RNO_POSITIVE_WEIGHT=${RNO_POSITIVE_WEIGHT:-0.0}

METHOD_TAG="MGR_RNO_${GENERATION_MODE}_seed${SEED}_lr${RNO_LR}-optsteps${RNO_STEPS}-rnogamma${RNO_GAMMA}-repelweight${RNO_REPEL_WEIGHT}-pos${RNO_POSITIVE_WEIGHT}"

OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/${DATASET}"}
IMAGE_ROOT=${IMAGE_ROOT:-"${OUTPUT_ROOT}/${METHOD_TAG}_step${STEP_START_MGR}_dump"}
MASK_OUTPUT_ROOT=${MASK_OUTPUT_ROOT:-"${OUTPUT_ROOT}/${METHOD_TAG}_step${STEP_START_MGR}_dump_sam3_masks"}

echo "[SAM3] Extract target masks"
echo "  DATASET=${DATASET}"
echo "  METHOD_TAG=${METHOD_TAG}"
echo "  IMAGE_ROOT=${IMAGE_ROOT}"
echo "  MASK_OUTPUT_ROOT=${MASK_OUTPUT_ROOT}"
echo "  POLICY_CSV=${POLICY_CSV}"
echo "  POLICY_DATASET=${DATASET}"
echo "  PROMPT=${SAM3_PROMPT}"
echo "  GPU_ID=${GPU_ID}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[ERROR] SAM3 extraction script not found: ${SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${POLICY_CSV}" ]]; then
  echo "[ERROR] policy manifest not found: ${POLICY_CSV}" >&2
  exit 1
fi

if [[ ! -d "${IMAGE_ROOT}" ]]; then
  echo "[ERROR] IMAGE_ROOT does not exist: ${IMAGE_ROOT}" >&2
  echo "Run Stage 1 first with matching arguments and hyperparameters." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" "${SCRIPT}" \
  --image_root "${IMAGE_ROOT}" \
  --output_root "${MASK_OUTPUT_ROOT}" \
  --policy_manifest_csv "${POLICY_CSV}" \
  --policy_dataset "${DATASET}" \
  --prompt "${SAM3_PROMPT}" \
  --overwrite

cat <<MSG

[SAM3 done]
Target mask root:
  ${MASK_OUTPUT_ROOT}

Next step:
  Run Stage 2 resume sampling with the same DATA/IMB_DATA_RATIO/SEED.
MSG
