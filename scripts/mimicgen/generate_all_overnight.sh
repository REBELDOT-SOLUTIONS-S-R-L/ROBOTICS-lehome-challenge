#!/usr/bin/env bash
# Sequentially generate mimicgen datasets for multiple garments, each with
# its own source teleop HDF5.  Each run's stdout+stderr is tee'd to a
# per-garment log.  If one run fails, the script continues with the next
# garment and reports a summary at the end.
# nohup ./scripts/mimicgen/generate_all_overnight.sh
set -u

INPUT_DIR="Datasets/hdf5_mimicgen_pipeline/1_annotated_teleop/Top_Long"
OUTPUT_DIR="Datasets/hdf5_mimicgen_pipeline/2_generated/Top_Long"
LOG_DIR="logs/mimicgen_overnight"
NUM_TRIALS="${NUM_TRIALS:-200}"

# Ordered garment list drives execution order.
GARMENTS=(
    Top_Long_Seen_0
    Top_Long_Seen_1
    Top_Long_Seen_2
    Top_Long_Seen_5
    Top_Long_Seen_7
    Top_Long_Seen_9
)

# Per-garment source teleop file (basename relative to $INPUT_DIR).
declare -A GARMENT_INPUTS=(
    [Top_Long_Seen_0]="Top_Long_Seen_0-HALTON_64-run_2.hdf5"
    [Top_Long_Seen_1]="Top_Long_Seen_0+5-HALTON_64.hdf5"
    [Top_Long_Seen_2]="Top_Long_Seen_0+5-HALTON_64.hdf5"
    [Top_Long_Seen_5]="Top_Long_Seen_5-HALTON_64.hdf5"
    [Top_Long_Seen_7]="Top_Long_Seen_0+5-HALTON_64.hdf5"
    [Top_Long_Seen_9]="Top_Long_Seen_0-HALTON_64-run_2.hdf5"
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_ACTIVATE="${REPO_ROOT}/.venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "ERROR: venv not found at $VENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"
echo "Using python: $(command -v python)"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Fail fast: every referenced input must exist before we start the overnight run.
for garment in "${GARMENTS[@]}"; do
    input_basename="${GARMENT_INPUTS[$garment]:-}"
    if [[ -z "$input_basename" ]]; then
        echo "ERROR: no input mapping for garment: $garment" >&2
        exit 1
    fi
    if [[ ! -f "${INPUT_DIR}/${input_basename}" ]]; then
        echo "ERROR: input file not found: ${INPUT_DIR}/${input_basename}" >&2
        exit 1
    fi
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="$LOG_DIR/summary_${TIMESTAMP}.log"
declare -a RESULTS=()

echo "Starting overnight generation at $(date)" | tee "$SUMMARY_LOG"
echo "Trials:   $NUM_TRIALS per garment" | tee -a "$SUMMARY_LOG"
echo "Mappings (garment -> input):" | tee -a "$SUMMARY_LOG"
for garment in "${GARMENTS[@]}"; do
    echo "  $garment -> ${GARMENT_INPUTS[$garment]}" | tee -a "$SUMMARY_LOG"
done
echo "----" | tee -a "$SUMMARY_LOG"

for garment in "${GARMENTS[@]}"; do
    input_basename="${GARMENT_INPUTS[$garment]}"
    input_file="${INPUT_DIR}/${input_basename}"
    # Output name: "<garment>-generated-<input-stem-without-garment-prefix>.hdf5"
    input_stem="${input_basename%.hdf5}"
    # Strip a leading "Top_Long_Seen_<anything>-" so outputs aren't double-labeled.
    output_suffix="${input_stem#Top_Long_Seen_*-}"
    output_file="${OUTPUT_DIR}/${garment}-generated-${output_suffix}.hdf5"
    run_log="${LOG_DIR}/${garment}_${TIMESTAMP}.log"

    echo "" | tee -a "$SUMMARY_LOG"
    echo "[$(date)] >>> Starting $garment" | tee -a "$SUMMARY_LOG"
    echo "    input:  $input_file" | tee -a "$SUMMARY_LOG"
    echo "    output: $output_file" | tee -a "$SUMMARY_LOG"
    echo "    log:    $run_log" | tee -a "$SUMMARY_LOG"

    start_ts=$SECONDS
    python scripts/mimicgen/generate_dataset.py \
        --task LeHome-BiSO101-ManagerBased-Garment-Mimic-v0 \
        --garment_name "$garment" \
        --input_file "$input_file" \
        --output_file "$output_file" \
        --pose_sequence "$NUM_TRIALS" \
        --device cuda \
        --num_envs 1 \
        --enable_cameras \
        --enable_pinocchio \
        2>&1 | tee "$run_log"
    status=${PIPESTATUS[0]}
    elapsed=$((SECONDS - start_ts))

    if [[ $status -eq 0 ]]; then
        echo "[$(date)] <<< OK  $garment (${elapsed}s)" | tee -a "$SUMMARY_LOG"
        RESULTS+=("OK   $garment (${elapsed}s)")
    else
        echo "[$(date)] <<< FAIL $garment (status=$status, ${elapsed}s)" | tee -a "$SUMMARY_LOG"
        RESULTS+=("FAIL $garment (status=$status, ${elapsed}s)")
    fi
done

echo "" | tee -a "$SUMMARY_LOG"
echo "==== Summary ====" | tee -a "$SUMMARY_LOG"
for line in "${RESULTS[@]}"; do
    echo "  $line" | tee -a "$SUMMARY_LOG"
done
echo "Finished at $(date)" | tee -a "$SUMMARY_LOG"
