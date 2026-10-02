"""Vector entity linking via Milvus (no local model needed).

Mirrors the project's KnowledgeGraphVectorStore collections:
  unconsolidated_entities_v1 (128) / unconsolidated_predicates_v1 (128)
  canonical_seed_entities_v1 (128) / canonical_predicates_v1 (128)
  raw_triples_v1 (256) / source_chunks_v1 (1024)
COSINE HNSW (M=16, efConstruction=200); search ef=max(64, limit).

Chain: extracted text -> jina embedding API (1024-dim, Matryoshka-truncate
+ renormalize to collection dim) -> Milvus -> PG entity id.
Milvus ids ARE Postgres entities.id (same uuids, verified).

Secrets/URLs from env only (never hardcoded):
  EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL (default jina)
  MILVUS_URI, MILVUS_TOKEN
  MILVUS_ENTITY_COLLECTION (default canonical_seed_entities_v1)
"""
import json
import math
import os
import urllib.request

_client = None

def _emb_base():
    return os.environ.get("EMBEDDING_BASE_URL", "http://172.25.80.168:8005/v1")

def _emb_key():
    k = os.environ.get("EMBEDDING_API_KEY", "")
    if not k:
        raise ValueError("Set env EMBEDDING_API_KEY first.")
    return k

def embed(texts, model=None):
    model = model or os.environ.get("EMBEDDING_MODEL", "jina")
    req = urllib.request.Request(
        _emb_base() + "/embeddings",
        data=json.dumps({"model": model, "input": texts}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + _emb_key()})
    res = json.load(urllib.request.urlopen(req, timeout=120))
    return [d["embedding"] for d in sorted(res["data"], key=lambda x: x["index"])]

def to_128(vec):
    tr = vec[:128]
    n = math.sqrt(sum(x * x for x in tr)) or 1.0
    return [x / n for x in tr]

def _client_or_raise():
    global _client
    if _client is not None:
        return _client
    from pymilvus import MilvusClient
    _client = MilvusClient(
        uri=os.environ.get("MILVUS_URI", "http://172.25.80.168:19530"),
        token=os.environ.get("MILVUS_TOKEN", "root:Milvus"))
    return _client

COLLECTIONS = {
    "unconsolidated_entities_v1": {"dim": 128, "fields": ("text", "entity_type",
        "node_id", "extraction_set")},
    "unconsolidated_predicates_v1": {"dim": 128, "fields": ("text",)},
    "canonical_seed_entities_v1": {"dim": 128, "fields": ("text", "canonical_name",
        "entity_type")},
    "canonical_predicates_v1": {"dim": 128, "fields": ("text", "canonical_name")},
    "raw_triples_v1": {"dim": 256, "fields": ("text", "subject_text",
        "predicate", "object_text")},
    "source_chunks_v1": {"dim": 1024, "fields": ("text", "node_id")},
}

def _default_collection():
    return os.environ.get("MILVUS_ENTITY_COLLECTION",
                           "canonical_seed_entities_v1")

def to_dim(vec, dim):
    tr = vec[:dim]
    n = math.sqrt(sum(x * x for x in tr)) or 1.0
    return [x / n for x in tr]

def assert_dimension(collection, expected):
    c = _client_or_raise()
    desc = c.describe_collection(collection)
    actual = int(next(f for f in desc["fields"]
                      if f["name"] == "vector")["params"]["dim"])
    if actual != expected:
        raise ValueError(f"{collection!r} dim {actual} != {expected}")
    return actual

def search_entities(texts, top_k=3, collection=None):
    """texts -> hit-lists [{id, text, distance, ...per-collection fields}]."""
    collection = collection or _default_collection()
    dim = COLLECTIONS[collection]["dim"]
    fields = list(COLLECTIONS[collection]["fields"])
    vecs = [to_dim(v, dim) for v in embed(texts)]
    c = _client_or_raise()
    c.load_collection(collection)
    out = c.search(collection, data=vecs, anns_field="vector", limit=top_k,
                   output_fields=fields,
                   search_params={"metric_type": "COSINE",
                                  "params": {"ef": max(64, top_k)}})
    res = []
    for hits in out:
        res.append([{"id": h.get("id"), "distance": h["distance"],
                     **{f: h["entity"].get(f) for f in fields}} for h in hits])
    return res

def search_predicates(texts, top_k=3,
                      collection="canonical_predicates_v1"):
    return search_entities(texts, top_k=top_k, collection=collection)

MAIN_TYPES = {"symptom", "concept", "disorder", "behavior"}
MIN_SIM = 0.8  # Milvus COSINE similarity floor; below it the hit is a guess


def _pg_exact_main(name):
    """Exact (case-insensitive) PG entity of a main type, else None."""
    try:
        import postgres_func as PG
    except Exception:
        return None
    try:
        c = PG._conn()
        try:
            cur = c.cursor()
            cur.execute("SELECT id, text, type FROM entities WHERE lower(text)=lower(%s)",
                        (name.strip(),))
            mains = [(str(i), t) for i, t, y in cur.fetchall()
                     if y in MAIN_TYPES]
            return mains[0] if mains else None
        finally:
            PG._put(c)
    except Exception:
        return None


def ranked(names, top_k=3, collection=None, min_sim=MIN_SIM):
    """Per-name ranked candidates: PG exact main-type hit first, then
    Milvus hits above min_sim (main types only), else best Milvus hit.
    Each candidate: {id, text, distance, src} where src is pg-exact,
    milvus or milvus-low."""
    names = [n.strip() for n in names if n and n.strip()]
    if not names:
        return []
    hit_lists = search_entities(names, top_k=top_k, collection=collection)
    out = []
    for n, hits in zip(names, hit_lists):
        cands = []
        exact = _pg_exact_main(n)
        if exact:
            cands.append({"id": exact[0], "text": exact[1],
                          "distance": 1.0, "src": "pg-exact"})
        for h in hits:
            if h.get("entity_type", h.get("type")) not in MAIN_TYPES:
                continue
            if h["distance"] < min_sim:
                continue
            cands.append({"id": h["id"], "text": h["text"],
                          "distance": h["distance"], "src": "milvus"})
        if not cands and hits:
            cands.append({"id": hits[0]["id"], "text": hits[0]["text"],
                          "distance": hits[0]["distance"], "src": "milvus-low"})
        out.append(cands[:top_k])
    return out


def link(names, top_k=3, collection=None, min_sim=MIN_SIM):
    """Ranked candidates flattened to ToG topic_entity (dedupes text)."""
    names = [n.strip() for n in names if n and n.strip()]
    if not names:
        return {}
    topic, seen_text = {}, set()
    for cands in ranked(names, top_k=top_k, collection=collection,
                        min_sim=min_sim):
        for h in cands:
            key = (h["text"] or "").strip().lower()
            if key and key not in seen_text and h["id"] not in topic:
                seen_text.add(key)
                topic[h["id"]] = h["text"]
    return topic

def as_link_fn(**kw):
    def _fn(name):
        t = link([name], **kw)
        if not t:
            return None
        k = next(iter(t))
        return (k, t[k])
    return _fn
