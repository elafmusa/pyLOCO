#!/bin/bash
#SBATCH --job-name=P4_TN_S1_preflight
#SBATCH --partition=petra4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/tracking_numerical_stage1_preflight_%j.out
#SBATCH --error=logs/tracking_numerical_stage1_preflight_%j.err

set -euo pipefail

P4_DIR="/data/dust/user/musa/beegfs.migration/PETRAIV"
PYTHON="/data/dust/user/musa/beegfs.migration/pyloco_latest_env/bin/python"
SEED="${P4_DIR}/pre-loco/pySC_petra4_06_seed111.json"
MEASURED="${P4_DIR}/loco_latest_pyloco_test/pySC_petra4_06_seed111/machine_cycle_01/measured_inputs.npz"
OUTPUT="${P4_DIR}/calculator_campaign/seed111_stage1_preflight/tracking_numerical_${SLURM_JOB_ID}"

cd "${P4_DIR}"
if [[ -e "${OUTPUT}" ]]; then
    echo "Refusing to overwrite existing output: ${OUTPUT}" >&2
    exit 2
fi

echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Output: ${OUTPUT}"

"${PYTHON}" run_p4_tracking_stage1_validation.py \
    --seed-file "${SEED}" \
    --measured "${MEASURED}" \
    --output "${OUTPUT}"

echo "End: $(date --iso-8601=seconds)"
echo "Summary: ${OUTPUT}/validation_summary.json"
