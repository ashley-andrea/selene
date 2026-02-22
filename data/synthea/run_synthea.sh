#!/usr/bin/env bash
# ============================================================
#  run_synthea.sh
#  Generates the synthetic women patient dataset for the
#  Birth Control Recommendation project.
#
#  Usage:
#    ./run_synthea.sh [population_size] [seed]
#
#  Defaults: 5000 patients, seed 42
#
#  Requirements: Java 11+ installed, Synthea cloned at ~/synthea
# ============================================================

set -euo pipefail

SYNTHEA_DIR="${SYNTHEA_DIR:-$HOME/synthea}"
CUSTOM_MODULES_DIR="$(cd "$(dirname "$0")/custom_modules" && pwd)"
CONFIG_FILE="$(cd "$(dirname "$0")" && pwd)/synthea.properties"
OUTPUT_DIR="$(cd "$(dirname "$0")/../.." && pwd)/data/output/patients"
POPULATION="${1:-5000}"
SEED="${2:-42}"

echo "============================================"
echo "  Synthea Synthetic Patient Generator"
echo "  Birth Control Recommendation Dataset"
echo "============================================"
echo "  Synthea dir  : $SYNTHEA_DIR"
echo "  Custom modules: $CUSTOM_MODULES_DIR"
echo "  Config file  : $CONFIG_FILE"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Population   : $POPULATION women"
echo "  Random seed  : $SEED"
echo "============================================"

# Verify Synthea is present
if [ ! -f "$SYNTHEA_DIR/run_synthea" ]; then
    echo "ERROR: Synthea not found at $SYNTHEA_DIR"
    echo "  Clone it with: git clone https://github.com/synthetichealth/synthea.git ~/synthea"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd "$SYNTHEA_DIR"

./run_synthea \
    -g F \
    -a 15-50 \
    -p "$POPULATION" \
    -s "$SEED" \
    -c "$CONFIG_FILE" \
    -d "$CUSTOM_MODULES_DIR" \
    --exporter.baseDirectory="$OUTPUT_DIR" \
    --exporter.csv.export=true \
    --exporter.fhir.export=false \
    Massachusetts

echo ""
echo "============================================"
echo "  Generation complete!"
echo "  Output CSV files in: $OUTPUT_DIR"
echo ""
echo "  Key files for our pipeline:"
echo "    patients.csv     -> demographics, income, education"
echo "    conditions.csv   -> diagnoses / contraindications"
echo "    observations.csv -> vitals (BMI, blood pressure)"
echo "    medications.csv  -> contraceptive history"
echo "============================================"
echo ""
echo "  Next step: run the post-processing script:"
echo "    python data/processing/flatten_patients.py \\"
echo "      --input $OUTPUT_DIR/csv \\"
echo "      --output data/output/patients_flat.csv"
