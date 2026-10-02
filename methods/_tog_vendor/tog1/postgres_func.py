"""ToG KG layer on DSM minimal Postgres (dsm5_minimal_graph).

Tables: entities(id uuid, text, type) 33,171
        facts(id, subject_id, predicate, object_id) 76,442
        mentions(fact_id, chunk_id) / chunks(id, content)
Entity id = entities.id. DSN from env DSM_KG_DATABASE_URL only.
"""
import os

_pool = None

def _pool_or_raise():
    global _pool
    if _pool is not None:
        return _pool
    try:
        import psycopg2.pool
    except ModuleNotFoundError:
        # The project uses psycopg v3. Keep the small getconn/putconn
        # interface expected by this vendored ToG code without adding the
        # obsolete psycopg2 dependency.
        import psycopg

        class _Psycopg3Pool:
            def __init__(self, dsn: str):
                self.dsn = dsn

            def getconn(self):
                return psycopg.connect(self.dsn)

            def putconn(self, connection):
                connection.close()

        dsn = os.environ.get("DSM_KG_DATABASE_URL", "")
        if not dsn:
            raise ValueError("Set env DSM_KG_DATABASE_URL first.")
        _pool = _Psycopg3Pool(dsn)
        return _pool
    dsn = os.environ.get("DSM_KG_DATABASE_URL", "")
    if not dsn:
        raise ValueError("Set env DSM_KG_DATABASE_URL first.")
    _pool = psycopg2.pool.SimpleConnectionPool(1, 5, dsn=dsn)
    return _pool

def _conn():
    return _pool_or_raise().getconn()

def _put(c):
    _pool_or_raise().putconn(c)

def link_entity(text):
    c = _conn()
    try:
        cur = c.cursor()
        cur.execute("SELECT id, text FROM entities WHERE lower(text)=lower(%s) LIMIT 1",
                    (text.strip(),))
        r = cur.fetchone()
        if r:
            return (str(r[0]), r[1])
        cur.execute("SELECT id, text FROM entities WHERE text ILIKE %s LIMIT 1",
                    ('%' + text.strip() + '%',))
        r = cur.fetchone()
        return (str(r[0]), r[1]) if r else None
    finally:
        _put(c)

def id2entity_name_or_type(entity_id):
    c = _conn()
    try:
        cur = c.cursor()
        cur.execute("SELECT text FROM entities WHERE id=%s", (str(entity_id),))
        r = cur.fetchone()
        return r[0] if r else "UnName_Entity"
    finally:
        _put(c)

def get_relations(entity_id):
    c = _conn()
    try:
        cur = c.cursor()
        cur.execute("SELECT DISTINCT predicate FROM facts WHERE subject_id=%s",
                    (str(entity_id),))
        head = [x[0] for x in cur.fetchall() if x[0]]
        cur.execute("SELECT DISTINCT predicate FROM facts WHERE object_id=%s",
                    (str(entity_id),))
        tail = [x[0] for x in cur.fetchall() if x[0]]
        return head, tail
    finally:
        _put(c)

def _direct(entity_id, relation, head, limit):
    """Raw 1-hop rows: [(id, text, type)]."""
    c = _conn()
    try:
        cur = c.cursor()
        if head:
            cur.execute(
                "SELECT f.object_id, e.text, e.type FROM facts f "
                "LEFT JOIN entities e ON e.id=f.object_id "
                "WHERE f.subject_id=%s AND f.predicate=%s LIMIT %s",
                (str(entity_id), relation, limit))
        else:
            cur.execute(
                "SELECT f.subject_id, e.text, e.type FROM facts f "
                "LEFT JOIN entities e ON e.id=f.subject_id "
                "WHERE f.object_id=%s AND f.predicate=%s LIMIT %s",
                (str(entity_id), relation, limit))
        return [(str(a), b if b else str(a), t) for a, b, t in cur.fetchall()]
    finally:
        _put(c)


MAIN_TYPES = {"symptom", "concept", "disorder", "behavior"}
"""Main nodes: walk stops here, they fill the beam. Every other type
(criterion, specifier, duration, patient, example_label, code,
unresolved, section_title) is an intermediate node resolved THROUGH."""


def _pass_facts(node_id, per_side=5):
    """All facts touching a node: [(rel, other_id, other_text,
    other_type, node_is_subject)]."""
    c = _conn()
    try:
        cur = c.cursor()
        cur.execute(
            "SELECT f.predicate, f.object_id, e.text, e.type FROM facts f "
            "LEFT JOIN entities e ON e.id=f.object_id "
            "WHERE f.subject_id=%s LIMIT %s", (str(node_id), per_side))
        out = [(r, str(a), b if b else str(a), t, True)
               for r, a, b, t in cur.fetchall()]
        cur.execute(
            "SELECT f.predicate, f.subject_id, e.text, e.type FROM facts f "
            "LEFT JOIN entities e ON e.id=f.subject_id "
            "WHERE f.object_id=%s LIMIT %s", (str(node_id), per_side))
        out += [(r, str(a), b if b else str(a), t, False)
                for r, a, b, t in cur.fetchall()]
        return out
    finally:
        _put(c)


def _walk_pass(node_id, node_text, seen, level, fanout):
    """Resolve intermediate node(s) to main-type endpoints.
    Returns [(endpoint_id, endpoint_text, trips, pass_ids)] where trips
    are ordered (subj, rel, obj, dir) tuples from this node outward."""
    res = []
    for rel, oid, otext, otype, node_subj in _pass_facts(node_id):
        if oid in seen:
            continue
        t = ((node_text, rel, otext, "out") if node_subj
             else (otext, rel, node_text, "in"))
        if otype in MAIN_TYPES:
            res.append((oid, otext, [t], [str(node_id)]))
        elif level < 1:
            for eid, etext, trips2, pids in _walk_pass(
                    oid, otext, seen | {oid}, level + 1, 3):
                res.append((eid, etext, [t] + trips2,
                            [str(node_id)] + pids))
        if len(res) >= fanout:
            break
    return res


def get_neighbors(entity_id, relation, head=True, limit=50,
                  resolve_pass=True):
    """1-hop neighbors. Intermediate-type hits are resolved THROUGH inline:
    returns [(id, text, via)] where via is None (main node) or
    {"entry", "trips", "pass_ids"} with the full sub-path from the
    intermediate entry to a main-type endpoint.
    Intermediates never appear as results, cost no depth (resolved inside
    this call); caller must visited-exclude pass_ids and use trips."""
    rows = _direct(entity_id, relation, head, limit)
    if not resolve_pass:
        return [(i, t, None) for i, t, _ in rows]
    out = []
    for nid, ntext, ntype in rows:
        if ntype in MAIN_TYPES:
            out.append((nid, ntext, None))
            continue
        seen = {str(entity_id), nid}
        for eid, etext, trips, pids in _walk_pass(nid, ntext, seen, 0, 6):
            if eid == str(entity_id):
                continue
            out.append((eid, etext, {"entry": ntext, "trips": trips,
                                     "pass_ids": pids}))
            if len(out) >= limit:
                break
    return out[:limit]
