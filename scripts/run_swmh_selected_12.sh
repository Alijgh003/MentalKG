#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

# User supplied human row numbers (1-based); run_method expects zero-based indices.
ROWS="10,16,24,36,38,41,51,71,76,79,81,82,88,104,113,118,129,136,142,167,169,207,227,229,230,238,247,253,261,262,263,265,278,289,292,300,313,325,328,345,356,357,368,369,383,395,403,410,415,427,466,468,473,485,513,532,533,540,542,548,549,574,579,584,595"
OUT="outputs/method_runs"
mkdir -p "$OUT"

run_one() {
  local method="$1" model_tag="$2" model="$3" provider="$4" label="$5"
  local out="$OUT/SWMH-rows65-${method}-${label}-${model_tag}.jsonl"
  local label_queries=""
  if [[ "$method" == "hipporag2" ]]; then
    [[ "$label" == "labelON" ]] && label_queries=true || label_queries=false
  fi
  echo "[$(date -Is)] START $method $model_tag $label -> $out"
  if [[ "$provider" == "morph" ]]; then
    LLM_MODEL="$model" LLM_PROVIDER_ONLY="morph" LLM_PROVIDER_ALLOW_FALLBACKS=false HIPPORAG2_USE_LABEL_FACT_QUERIES="$label_queries" \
      HIPPORAG2_QA_TOP_K=8 HIPPORAG2_RETRIEVAL_TOP_K=8 \
      VANILLA_RAG_QA_TOP_K=8 VANILLA_RAG_RETRIEVAL_TOP_K=8 \
      .venv/bin/python scripts/run_method.py --method "$method" --dataset SWMH --split test \
      --row-indices "$ROWS" --generate-answer --output "$out" \
      >/dev/null 2>&1
  else
    LLM_MODEL="$model" LLM_PROVIDER_ONLY="" LLM_PROVIDER_ALLOW_FALLBACKS="" HIPPORAG2_USE_LABEL_FACT_QUERIES="$label_queries" \
      HIPPORAG2_QA_TOP_K=8 HIPPORAG2_RETRIEVAL_TOP_K=8 \
      VANILLA_RAG_QA_TOP_K=8 VANILLA_RAG_RETRIEVAL_TOP_K=8 \
      .venv/bin/python scripts/run_method.py --method "$method" --dataset SWMH --split test \
      --row-indices "$ROWS" --generate-answer --output "$out" >/dev/null 2>&1
  fi
  echo "[$(date -Is)] END $method $model_tag $label rc=$?"
}

# Launch one job for each method/model/configuration.
for spec in \
  "hipporag2 gemma4-31b openrouter/google/gemma-4-31b-it:nitro none labelON" \
  "hipporag2 gemma4-31b openrouter/google/gemma-4-31b-it:nitro none labelOFF" \
  "vanilla_rag gemma4-31b openrouter/google/gemma-4-31b-it:nitro none none" \
  "cot gemma4-31b openrouter/google/gemma-4-31b-it:nitro none none" \
  "hipporag2 deepseek-v4.1-flash deepseek/deepseek-v4.1-flash morph labelON" \
  "hipporag2 deepseek-v4.1-flash deepseek/deepseek-v4.1-flash morph labelOFF" \
  "vanilla_rag deepseek-v4.1-flash deepseek/deepseek-v4.1-flash morph none" \
  "cot deepseek-v4.1-flash deepseek/deepseek-v4.1-flash morph none" \
  "hipporag2 gemma3-4b openrouter/google/gemma-3-4b-it none labelON" \
  "hipporag2 gemma3-4b openrouter/google/gemma-3-4b-it none labelOFF" \
  "vanilla_rag gemma3-4b openrouter/google/gemma-3-4b-it none none" \
  "cot gemma3-4b openrouter/google/gemma-3-4b-it none none"; do
  read -r method tag model provider label <<< "$spec"
  run_one "$method" "$tag" "$model" "$provider" "$label" &
done

wait
echo "[$(date -Is)] ALL JOBS FINISHED"
