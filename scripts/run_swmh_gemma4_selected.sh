#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
ROWS="10,16,24,36,38,41,51,71,76,79,81,82,88,104,113,118,129,136,142,167,169,207,227,229,230,238,247,253,261,262,263,265,278,289,292,300,313,325,328,345,356,357,368,369,383,395,403,410,415,427,466,468,473,485,513,532,533,540,542,548,549,574,579,584,595"
OUT="outputs/method_runs"
mkdir -p "$OUT"
run_one() {
  local method="$1" tag="$2"
  local out="$OUT/SWMH-rows65-${method}-${tag}-gemma4-31b.jsonl"
  echo "[$(date -Is)] START $method $tag"
  if [[ "$method" == hipporag2 ]]; then
    HIPPORAG2_USE_LABEL_FACT_QUERIES="$([[ "$tag" == labelON ]] && echo true || echo false)" \
      HIPPORAG2_QA_TOP_K=8 HIPPORAG2_RETRIEVAL_TOP_K=8 \
      .venv/bin/python scripts/run_method.py --method "$method" --dataset SWMH --split test \
      --row-indices "$ROWS" --generate-answer --output "$out"
  else
    VANILLA_RAG_QA_TOP_K=8 VANILLA_RAG_RETRIEVAL_TOP_K=8 \
      .venv/bin/python scripts/run_method.py --method "$method" --dataset SWMH --split test \
      --row-indices "$ROWS" --generate-answer --output "$out"
  fi
  rc=$?
  echo "[$(date -Is)] END $method $tag rc=$rc"
  return $rc
}
run_one hipporag2 labelON &
run_one hipporag2 labelOFF &
run_one vanilla_rag none &
run_one cot none &
wait
echo "[$(date -Is)] ALL JOBS FINISHED"
