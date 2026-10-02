const state = { run: null, sampleIndex: 0, stage: null };
const $ = (id) => document.getElementById(id);

const stageNames = {
  fact_generation: "Generated triples",
  fact_retrieval: "Retrieved facts",
  passage_retrieval: "Dense chunks",
  reranking: "Reranked chunks",
  seed_weighting: "Entity seeds",
  ppr: "PPR nodes",
  passage_ranking: "Final chunks",
  cot_generation: "CoT reasoning",
  answer_generation: "Final answer",
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}

function shortId(value) { return value ? `${value.slice(0, 8)}…` : "—"; }
function fmt(value, digits = 3) { return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—"; }

async function loadRuns() {
  const runs = await fetch("/api/runs").then(response => response.json());
  const select = $("run-select");
  select.innerHTML = runs.map(run => `<option value="${escapeHtml(run.name)}">${escapeHtml(run.name)}</option>`).join("");
  if (runs.length) await loadRun(runs[0].name);
  else showError("No run JSONL files were found in outputs/method_runs.");
}

async function loadRun(name) {
  $("empty-state").hidden = false;
  $("sample-view").hidden = true;
  $("empty-state").innerHTML = `<div class="spinner"></div><h2>Loading run</h2>`;
  const payload = await fetch(`/api/run?file=${encodeURIComponent(name)}`).then(response => response.json());
  if (payload.error) return showError(payload.error);
  state.run = payload;
  state.sampleIndex = 0;
  renderSamples();
  if (payload.samples.length) selectSample(0);
  else showError("This run contains no samples.");
}

function renderSamples() {
  const samples = state.run.samples;
  $("sample-count").textContent = `${samples.length} sample${samples.length === 1 ? "" : "s"}`;
  $("sample-list").innerHTML = samples.map((sample, index) => {
    const generated = sample.stage_outputs?.fact_generation?.length ?? 0;
    const facts = sample.stage_outputs?.fact_retrieval?.length ?? 0;
    return `<button class="sample-item ${index === state.sampleIndex ? "active" : ""}" data-index="${index}">
      <span class="sample-number">${String(index + 1).padStart(2, "0")}</span>
      <span class="sample-copy"><strong>Row ${sample.dataset_row_index ?? index}</strong><small>${escapeHtml(sample.sample_id)}</small></span>
      <span class="sample-result ${facts ? "has-result" : ""}">${generated}→${facts}</span>
    </button>`;
  }).join("");
  document.querySelectorAll(".sample-item").forEach(button => button.addEventListener("click", () => selectSample(Number(button.dataset.index))));
}

function selectSample(index) {
  state.sampleIndex = index;
  const sample = state.run.samples[index];
  const stages = Object.keys(sample.stage_outputs || {}).filter(key => stageNames[key]);
  state.stage = stages.includes(state.stage) ? state.stage : stages[0];
  renderSamples();
  $("empty-state").hidden = true;
  $("sample-view").hidden = false;
  $("sample-location").textContent = `${sample.dataset?.toUpperCase() || "DATASET"} / ${sample.split || "split"} · dataset row ${sample.dataset_row_index} · source row ${sample.source_row_index ?? "—"}`;
  $("sample-id").textContent = sample.sample_id;
  $("input-text").textContent = sample.input?.text || "No input text recorded.";
  const generated = sample.stage_outputs?.fact_generation?.length ?? 0;
  const facts = sample.stage_outputs?.fact_retrieval?.length ?? 0;
  const latency = sample.cost?.latency_ms;
  const gold = sample.gold?.label || "—";
  const predicted = sample.output?.labels?.[0] || "—";
  const match = predicted !== "—" && gold !== "—" && String(predicted).toLowerCase() === String(gold).toLowerCase();
  $("sample-stats").innerHTML = stat("Gold answer", gold) + stat("Model answer", predicted) + stat("Match", match ? "✓" : (predicted === "—" ? "—" : "✗")) + stat("Queries", generated) + stat("Facts", facts) + stat("Latency", latency == null ? "—" : `${(latency / 1000).toFixed(2)}s`);
  renderTabs(stages);
  renderStage();
}

function stat(label, value) { return `<div><span>${escapeHtml(value)}</span><small>${label}</small></div>`; }

function renderTabs(stages) {
  $("stage-tabs").innerHTML = stages.map((stage, index) => `<button class="stage-tab ${stage === state.stage ? "active" : ""}" data-stage="${stage}"><span>${index + 1}</span>${stageNames[stage]}</button>`).join("");
  document.querySelectorAll(".stage-tab").forEach(button => button.addEventListener("click", () => { state.stage = button.dataset.stage; renderTabs(stages); renderStage(); }));
}

function renderStage() {
  const sample = state.run.samples[state.sampleIndex];
  const data = sample.stage_outputs?.[state.stage];
  const content = $("stage-content");
  if (state.stage === "fact_generation") content.innerHTML = renderGenerated(data || []);
  else if (state.stage === "fact_retrieval") content.innerHTML = renderFacts(data || [], sample);
  else if (state.stage === "ppr") content.innerHTML = renderPpr(data || {});
  else if (state.stage === "seed_weighting") content.innerHTML = renderSeeds(data || []);
  else if (state.stage === "reranking") content.innerHTML = renderReranked(data || []);
  else if (state.stage === "cot_generation") content.innerHTML = renderCoT(data || {});
  else if (state.stage === "answer_generation") content.innerHTML = renderAnswer(data || {});
  else content.innerHTML = renderPassages(data || []);
}

function sectionIntro(title, copy, badge) {
  return `<div class="section-intro"><div><h3>${title}</h3><p>${copy}</p></div><span class="count-badge">${badge}</span></div>`;
}

function renderGenerated(rows) {
  if (!rows.length) return sectionIntro("No triples generated", "The model found no defensible clinical search facts in this sample.", "0") + emptyBox();
  return sectionIntro("LLM-generated search triples", "Each triple is embedded and searched independently against the Milvus fact collection.", `${rows.length} queries`) +
    `<div class="triple-grid">${rows.map(row => `<article class="triple-card"><span class="query-id">${escapeHtml(row.query_id)}</span><div class="triple-flow"><strong>${escapeHtml(row.subject)}</strong><span>${escapeHtml(row.predicate)}</span><strong>${escapeHtml(row.object)}</strong></div><code>${escapeHtml(row.text)}</code></article>`).join("")}</div>`;
}

function renderFacts(rows, sample) {
  const metrics = (sample.cost?.stages || []).find(item => item.stage === "fact_retrieval")?.details || {};
  const info = `<div class="metric-strip">${stat("Merged pool", metrics.unique_hits_before_final_k ?? "—")}${stat("Raw range", metrics.normalization_raw_min == null ? "—" : `${fmt(metrics.normalization_raw_min)}–${fmt(metrics.normalization_raw_max)}`)}${stat("Kept", rows.length)}</div>`;
  const queries = [
    ...(sample.stage_outputs?.fact_generation || []),
    ...(sample.stage_outputs?.label_fact_queries || []),
  ];
  const queryPanel = queries.length ? `<div class="query-panel"><span class="detail-label">Queries used for fact retrieval</span>${queries.map(query => `<div class="match"><b>${escapeHtml(query.query_id)}</b><span>${escapeHtml(query.text || `${query.subject} ${query.predicate} ${query.object}`)}</span>${query.query_type === "label_target" ? `<small>label target: ${escapeHtml(query.target_label || "")}</small>` : `<small>model-generated</small>`}</div>`).join("")}</div>` : "";
  if (!rows.length) return sectionIntro("No retrieved facts", "No query triples were generated, or no matching facts were returned.", "0 facts") + info + emptyBox();
  return sectionIntro("Facts retained for graph seeding", "Sorted after global min-max normalization of the deduplicated candidate pool. Open a row to see exactly which generated query matched it.", `${rows.length} facts`) + info +
    queryPanel + `<div class="fact-list">${rows.map(renderFact).join("")}</div>`;
}

function renderFact(row) {
  const matches = row.matched_queries || [];
  return `<details class="fact-row"><summary>
    <span class="rank">${row.rank}</span>
    <span class="fact-main"><span class="fact-triple"><strong>${escapeHtml(row.subject)}</strong><em>${escapeHtml(row.predicate)}</em><strong>${escapeHtml(row.object)}</strong></span><small>${shortId(row.triple_id)} · ${matches.length} query match${matches.length === 1 ? "" : "es"}</small></span>
    <span class="score-cell"><span class="score-value">${fmt(row.raw_score)}</span><small>raw cosine</small><span class="score-track"><i style="width:${Math.max(0, Math.min(100, Number(row.score) * 100))}%"></i></span><small>${fmt(row.score)} normalized</small></span>
  </summary><div class="fact-detail"><div><span class="detail-label">Database fact</span><code>${escapeHtml(row.text)}</code></div><div><span class="detail-label">Matched generated queries</span>${matches.map(match => `<div class="match"><b>fq${match.query_rank}</b><span>${escapeHtml(match.query_text)}</span><small>hit #${match.hit_rank} · cosine ${fmt(match.score)}</small></div>`).join("")}</div></div></details>`;
}

function renderSeeds(rows) {
  return sectionIntro("Entity restart weights", "Clinically seed-eligible fact endpoints after degree penalty.", `${rows.length} seeds`) + (rows.length ? `<div class="simple-list">${rows.map(row => `<div><b>#${row.rank} ${escapeHtml(row.type)}</b><span>${shortId(row.entity_id)}</span><strong>${fmt(row.score)}</strong></div>`).join("")}</div>` : emptyBox());
}

function renderPpr(data) {
  if (data.skipped) return sectionIntro("PPR skipped", data.reason || "No valid graph seeds were available.", "skipped") + emptyBox();
  const rows = data.top_nodes || [];
  return sectionIntro("Personalized PageRank", `Converged in ${data.iterations ?? "—"} iterations.`, `${rows.length} nodes`) + `<div class="simple-list">${rows.map((row, index) => `<div><b>#${index + 1}</b><span>${escapeHtml(row.node_id)}</span><strong>${fmt(row.score, 5)}</strong></div>`).join("")}</div>`;
}

function renderPassages(rows) {
  return sectionIntro(stageNames[state.stage] || "Stage output", "Retrieved source text ordered by relevance.", `${rows.length} chunks`) + (rows.length ? `<div class="passage-list">${rows.map(row => `<article><div><b>#${row.rank}</b><code>${shortId(row.passage_id)}</code><strong>${fmt(row.score, 5)}</strong></div><p>${escapeHtml(row.text || "No hydrated text recorded at this stage.")}</p></article>`).join("")}</div>` : emptyBox());
}

function renderReranked(rows) {
  return sectionIntro("Reranked chunks", "Dense candidates reordered by the configured reranking endpoint.", `${rows.length} chunks`) + (rows.length ? `<div class="passage-list">${rows.map(row => `<article><div><b>#${row.rank}</b><code>${shortId(row.passage_id)}</code><strong>${fmt(row.rerank_score, 5)} rerank</strong></div><small>dense #${row.dense_rank} · ${fmt(row.dense_score, 5)}</small><p>${escapeHtml(row.text || "No hydrated text recorded at this stage.")}</p></article>`).join("")}</div>` : emptyBox());
}

function renderCoT(data) {
  if (!data || !data.answer) return sectionIntro("CoT reasoning", "Chain-of-Thought without retrieval.", data.answer || "—") + emptyBox();
  return sectionIntro("CoT reasoning", "LLM-only ChainOfThought: post + allowed labels → answer + reasoning.", data.answer) +
    `<article class="triple-card"><span class="query-id">ANSWER</span><p><strong>${escapeHtml(data.answer)}</strong></p><span class="detail-label">Reasoning</span><p>${escapeHtml(data.reasoning || "No reasoning")}</p><span class="detail-label">Valid labels</span><p>${escapeHtml((data.valid_labels||[]).join(", "))}</p></article>`;
}

function renderAnswer(data) {
  const sample = state.run.samples[state.sampleIndex];
  const citations = data.cited_passage_ids || [];
  const gold = sample.gold?.label || "—";
  const predicted = data.answer || "—";
  const match = predicted !== "—" && gold !== "—" && String(predicted).toLowerCase() === String(gold).toLowerCase();
  return sectionIntro("Evidence-grounded answer", "Generated from the highest-ranked evidence passages.", predicted) +
    `<div class="metric-strip">${stat("Gold answer", gold)}${stat("Model answer", predicted)}${stat("Result", match ? "Correct" : "Different")}</div>` +
    `<article class="triple-card"><span class="query-id">MODEL EXPLANATION</span><p>${escapeHtml(data.explanation || "No explanation was returned.")}</p><span class="detail-label">Cited passages</span><div>${citations.length ? citations.map(id => `<code>${escapeHtml(id)}</code>`).join("<br>") : "No valid citations returned."}</div></article>`;
}

function emptyBox() { return `<div class="empty-box">Nothing to display at this stage.</div>`; }
function showError(message) { $("empty-state").hidden = false; $("sample-view").hidden = true; $("empty-state").innerHTML = `<div class="empty-glyph">!</div><h2>Could not load viewer</h2><p>${escapeHtml(message)}</p>`; }

$("run-select").addEventListener("change", event => loadRun(event.target.value));
loadRuns().catch(error => showError(error.message));
