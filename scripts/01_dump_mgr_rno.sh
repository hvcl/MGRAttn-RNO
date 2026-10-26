#!/usr/bin/env bash
set -euo pipefail

# Stage 1: run denoising up to the MGR start step and dump target images/latents.
# The dumped target images are used for SAM3 target-mask extraction.
#
# Supported datasets:
#   - CUB-LT IF100    : DATA=cub,    IMB_DATA_RATIO=0.01 -> DATASET=imb_cub_0.01
#   - Flower-LT IF100 : DATA=flower, IMB_DATA_RATIO=0.01 -> DATASET=imb_flower_0.01
#
# Usage:
#   bash scripts/01_dump_mgr_rno.sh cub 0.01 0 0 44
#   bash scripts/01_dump_mgr_rno.sh flower 0.01 0 0 44
#
# By default, this script samples only the selected demonstration
# classes below and uses an explicit per-class demo quota. This avoids the
# selected-tail-only normalization edge case where all selected 1-shot classes
# produce zero samples.
#
# To change the demo count:
#   DEMO_QUOTA_PER_CLASS=10 bash scripts/01_dump_mgr_rno.sh cub 0.01 0 0 44
#
# To override classes:
#   ONLY_CLASSES="Some_Class,Another_Class" bash scripts/01_dump_mgr_rno.sh cub 0.01 0 0 44
#
# To run the original full long-tailed normalization over all classes:
#   ONLY_CLASSES=all USE_DEMO_QUOTA=0 bash scripts/01_dump_mgr_rno.sh cub 0.01 0 0 44

export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export N_PROC=${N_PROC:-1}

DATA=${1:?"DATA is required, e.g., cub or flower"}
IMB_DATA_RATIO=${2:?"IMB_DATA_RATIO is required, e.g., 0.01"}
PROC_ID=${3:?"PROC_ID is required, e.g., 0"}
GPU_ID=${4:?"GPU_ID is required, e.g., 0"}
SEED=${5:-44}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

DATASET="imb_${DATA}_${IMB_DATA_RATIO}"

# ------------------------------------------------------------
# Dataset-specific default hyperparameters and selected classes
# ------------------------------------------------------------
case "${DATASET}" in
  "imb_cub_0.01")
    MGR_BLOCK_FROM=${MGR_BLOCK_FROM:-up2_2}
    RNO_LR=${RNO_LR:-0.01}
    RNO_STEPS=${RNO_STEPS:-3}
    RNO_REPEL_WEIGHT=${RNO_REPEL_WEIGHT:-1.0}
    DEFAULT_ONLY_CLASSES="Blue_winged_Warbler,Common_Yellowthroat,Le_Conte_Sparrow"
    ;;
  "imb_flower_0.01")
    MGR_BLOCK_FROM=${MGR_BLOCK_FROM:-up2_1}
    RNO_LR=${RNO_LR:-0.08}
    RNO_STEPS=${RNO_STEPS:-30}
    RNO_REPEL_WEIGHT=${RNO_REPEL_WEIGHT:-2.0}
    DEFAULT_ONLY_CLASSES="toad_lily,bearded_iris,blanket_flower"
    ;;
  *)
    echo "[ERROR] Unsupported dataset: ${DATASET}" >&2
    echo "Supported datasets are:" >&2
    echo "  - imb_cub_0.01    via: bash scripts/01_dump_mgr_rno.sh cub 0.01 ..." >&2
    echo "  - imb_flower_0.01 via: bash scripts/01_dump_mgr_rno.sh flower 0.01 ..." >&2
    exit 1
    ;;
esac

# ------------------------------------------------------------
# Common MGR / RNO hyperparameters
# ------------------------------------------------------------
STEP_START_MGR=${STEP_START_MGR:-20}
TARGET_BATCH_SIZE=${TARGET_BATCH_SIZE:-4}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7.0}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}

# Keep the paper/reproduction mode as normalization.
# For selected demonstration classes, class_quota_csv below provides the actual
# generation count to avoid zero samples from selected-tail-only normalization.
GENERATION_MODE=${GENERATION_MODE:-normalization}
EXPANSION_RATE=${EXPANSION_RATE:-5}
ONLY_CLASSES=${ONLY_CLASSES:-"${DEFAULT_ONLY_CLASSES}"}
MAX_CLASSES=${MAX_CLASSES:-0}

RNO_GAMMA=${RNO_GAMMA:-3.0}
RNO_POSITIVE_WEIGHT=${RNO_POSITIVE_WEIGHT:-0.0}
RNO_NEG_ANCHOR_EXTRA=${RNO_NEG_ANCHOR_EXTRA:-2}
RNO_POOL_SIZE=${RNO_POOL_SIZE:-16}

# Demo quota is used only when selected classes are used.
USE_DEMO_QUOTA=${USE_DEMO_QUOTA:-1}
DEMO_QUOTA_PER_CLASS=${DEMO_QUOTA_PER_CLASS:-50}

# ------------------------------------------------------------
# Repository-relative paths
# ------------------------------------------------------------
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
RUN_PY=${RUN_PY:-"${PROJECT_ROOT}/src/sampling/run_mgr_rno_sampling.py"}

DATA_ROOT=${DATA_ROOT:-"${PROJECT_ROOT}/data"}
TRAIN_DIR=${TRAIN_DIR:-"${DATA_ROOT}/${DATASET}/train"}

FT_DB_ROOT=${FT_DB_ROOT:-"${PROJECT_ROOT}/assets/ft_ti_db/${DATASET}"}
SUFFIX_ROOT=${SUFFIX_ROOT:-"${PROJECT_ROOT}/assets/suffix"}
REF_MASK_MANIFEST=${REF_MASK_MANIFEST:-"${PROJECT_ROOT}/assets/final_ref_mask_manifest.csv"}

OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/${DATASET}"}
METHOD_TAG="MGR_RNO_${GENERATION_MODE}_seed${SEED}_lr${RNO_LR}-optsteps${RNO_STEPS}-rnogamma${RNO_GAMMA}-repelweight${RNO_REPEL_WEIGHT}-pos${RNO_POSITIVE_WEIGHT}"
DUMP_ROOT=${DUMP_ROOT:-"${OUTPUT_ROOT}/${METHOD_TAG}_step${STEP_START_MGR}_dump"}

# Stage 1 is dump-only. Avoid creating a separate *_preview directory.
# The sampler requires --result_dir, so point it to DUMP_ROOT.
RESULT_DIR=${RESULT_DIR:-"${DUMP_ROOT}"}

CLASS_QUOTA_CSV=${CLASS_QUOTA_CSV:-""}

# If selected demonstration classes are used, create a small quota CSV so that
# tail-only selected classes still generate samples while the mode/tag remains
# "normalization".
if [[ -z "${CLASS_QUOTA_CSV}" && "${USE_DEMO_QUOTA}" == "1" && \
      -n "${ONLY_CLASSES}" && "${ONLY_CLASSES}" != "all" && "${ONLY_CLASSES}" != "ALL" ]]; then
  CLASS_QUOTA_CSV="${OUTPUT_ROOT}/demo_class_quota_${DATASET//\//_}_seed${SEED}.csv"
  mkdir -p "$(dirname "${CLASS_QUOTA_CSV}")"
  {
    echo "class,quota"
    IFS=',' read -ra _classes <<< "${ONLY_CLASSES}"
    for _cls in "${_classes[@]}"; do
      _cls="$(echo "${_cls}" | xargs)"
      if [[ -n "${_cls}" ]]; then
        echo "${_cls},${DEMO_QUOTA_PER_CLASS}"
      fi
    done
  } > "${CLASS_QUOTA_CSV}"
fi

EXTRA_ARGS=()
if [[ -n "${ONLY_CLASSES}" && "${ONLY_CLASSES}" != "all" && "${ONLY_CLASSES}" != "ALL" ]]; then
  EXTRA_ARGS+=(--only_classes "${ONLY_CLASSES}")
fi
if [[ -n "${CLASS_QUOTA_CSV}" ]]; then
  EXTRA_ARGS+=(--class_quota_csv "${CLASS_QUOTA_CSV}")
fi

echo "[Stage 1] Dump step-${STEP_START_MGR} targets"
echo "  DATASET=${DATASET}"
echo "  TRAIN_DIR=${TRAIN_DIR}"
echo "  FT_DB_ROOT=${FT_DB_ROOT}"
echo "  REF_MASK_MANIFEST=${REF_MASK_MANIFEST}"
echo "  GENERATION_MODE=${GENERATION_MODE}"
echo "  ONLY_CLASSES=${ONLY_CLASSES}"
echo "  USE_DEMO_QUOTA=${USE_DEMO_QUOTA}"
echo "  DEMO_QUOTA_PER_CLASS=${DEMO_QUOTA_PER_CLASS}"
echo "  CLASS_QUOTA_CSV=${CLASS_QUOTA_CSV:-<none>}"
echo "  MGR_BLOCK_FROM=${MGR_BLOCK_FROM}"
echo "  RNO_LR=${RNO_LR}"
echo "  RNO_STEPS=${RNO_STEPS}"
echo "  RNO_GAMMA=${RNO_GAMMA}"
echo "  RNO_REPEL_WEIGHT=${RNO_REPEL_WEIGHT}"
echo "  RNO_POSITIVE_WEIGHT=${RNO_POSITIVE_WEIGHT}"
echo "  DUMP_ROOT=${DUMP_ROOT}"
echo "  RESULT_DIR=${RESULT_DIR}  # no separate preview directory"

python "${RUN_PY}" \
  --train_dir "${TRAIN_DIR}" \
  --datasets "${DATASET}" \
  --shot 5shot \
  --ft_db_root "${FT_DB_ROOT}" \
  --model_id runwayml/stable-diffusion-v1-5 \
  --suffix_root "${SUFFIX_ROOT}" \
  --ref_mask_manifest "${REF_MASK_MANIFEST}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --generation_mode "${GENERATION_MODE}" \
  --expansion_rate "${EXPANSION_RATE}" \
  --max_classes "${MAX_CLASSES}" \
  "${EXTRA_ARGS[@]}" \
  --temperature 0.5 \
  --target_batch_size "${TARGET_BATCH_SIZE}" \
  --step_start_mgr "${STEP_START_MGR}" \
  --mgr_block_from "${MGR_BLOCK_FROM}" \
  --target_step_dump_root "${DUMP_ROOT}" \
  --dump_target_step_only \
  --mode sampling \
  --require_ref_seg \
  --ref_kv_cache_mode off \
  --N_proc "${N_PROC}" \
  --cur_proc "${PROC_ID}" \
  --use_mgr \
  --use_rno \
  --rno_negative_prompt_no_suffix \
  --rno_fixed_neg_anchor \
  --rno_exclude_self_neg \
  --rno_force_per_sample \
  --rno_num_opt_steps "${RNO_STEPS}" \
  --rno_lr "${RNO_LR}" \
  --rno_pool_size "${RNO_POOL_SIZE}" \
  --rno_ref_repel_weight "${RNO_REPEL_WEIGHT}" \
  --rno_gamma "${RNO_GAMMA}" \
  --rno_neg_anchor_extra "${RNO_NEG_ANCHOR_EXTRA}" \
  --rno_positive_weight "${RNO_POSITIVE_WEIGHT}" \
  --result_dir "${RESULT_DIR}" \
  --base_seed "${SEED}"

cat <<MSG

[Stage 1 done]
Dump root:
  ${DUMP_ROOT}

Next step:
  Run SAM3 target-mask extraction using the dumped step-${STEP_START_MGR} target images.
MSG
