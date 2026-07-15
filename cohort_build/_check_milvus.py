# -*- coding: utf-8 -*-
"""
_check_milvus.py — sanity check for CR-MSR M1.5 server-side execution.

Verifies:
  1) Milvus connection (host:port reachable)
  2) Three required collections exist
       - MedRAG_PubMed
       - MedRAG_Wikipedia
       - Medical_Guidelines
  3) Each collection is loaded (or loadable)
  4) Embedding dim is consistent (must equal embedder output dim)
  5) Schema has the expected 'vector' and 'content' fields
  6) A 1-query smoke search succeeds (returns top-3 with score)

Usage:
  python _check_milvus.py \
      --embed_model_path /mnt/.../gte-large-en-v1.5 \
      --milvus_host localhost --milvus_port 19530

Exit codes:
  0 = all checks passed
  1 = one or more checks failed (see stdout for details)
"""
import argparse
import os
import sys
import traceback

REQUIRED_COLLECTIONS = ["MedRAG_PubMed", "MedRAG_Wikipedia", "Medical_Guidelines"]
EMBED_FIELD = "vector"
TEXT_FIELD  = "content"
SEARCH_PARAMS = {"metric_type": "IP", "params": {"nprobe": 10}}


def _ok(msg: str):
    print(f"[OK]    {msg}")

def _fail(msg: str):
    print(f"[FAIL]  {msg}")

def _info(msg: str):
    print(f"[INFO]  {msg}")


def check_connection(host: str, port: int) -> bool:
    try:
        from pymilvus import connections
        connections.connect(alias="default", host=host, port=str(port))
        _ok(f"connected to Milvus at {host}:{port}")
        return True
    except Exception as e:
        _fail(f"Milvus connect: {e}")
        return False


def check_collections() -> dict:
    """Returns {coll_name: Collection or None}."""
    from pymilvus import utility, Collection
    out = {}
    for name in REQUIRED_COLLECTIONS:
        try:
            exists = utility.has_collection(name)
            if not exists:
                _fail(f"collection missing: {name}")
                out[name] = None
                continue
            col = Collection(name)
            _ok(f"collection found: {name}  (entities={col.num_entities})")
            out[name] = col
        except Exception as e:
            _fail(f"collection check {name}: {e}")
            out[name] = None
    return out


def check_schema(col, name: str) -> int:
    """Returns embedding dim if OK, -1 if failed."""
    try:
        fields = {f.name: f for f in col.schema.fields}
        if EMBED_FIELD not in fields:
            _fail(f"{name}: missing field '{EMBED_FIELD}'")
            return -1
        if TEXT_FIELD not in fields:
            _fail(f"{name}: missing field '{TEXT_FIELD}'")
            return -1
        dim = fields[EMBED_FIELD].params.get("dim", -1)
        if dim <= 0:
            _fail(f"{name}: invalid dim={dim}")
            return -1
        _ok(f"{name}: schema OK  ('{EMBED_FIELD}' dim={dim}, '{TEXT_FIELD}' present)")
        return int(dim)
    except Exception as e:
        _fail(f"{name}: schema check failed: {e}")
        return -1


def load_collection(col, name: str) -> bool:
    try:
        col.load()
        _ok(f"{name}: loaded into memory")
        return True
    except Exception as e:
        _fail(f"{name}: load failed: {e}")
        return False


def check_embedder(model_path: str) -> tuple:
    """Returns (embedder, dim) or (None, -1)."""
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_path, trust_remote_code=True)
        vec = m.encode(["sanity check query"], normalize_embeddings=True)
        dim = int(vec.shape[1])
        _ok(f"embedder loaded: {os.path.basename(model_path)}  (out_dim={dim})")
        return m, dim
    except Exception as e:
        _fail(f"embedder load failed: {e}")
        traceback.print_exc(limit=2)
        return None, -1


def smoke_search(col, embedder, name: str) -> bool:
    try:
        q = "patient with chest pain and shortness of breath"
        vec = embedder.encode([q], normalize_embeddings=True)
        res = col.search(
            data=[vec[0].tolist()],
            anns_field=EMBED_FIELD,
            param=SEARCH_PARAMS,
            limit=3,
            output_fields=[TEXT_FIELD],
        )
        hits = res[0]
        if len(hits) == 0:
            _fail(f"{name}: search returned 0 hits")
            return False
        top = hits[0]
        text_preview = (top.entity.get(TEXT_FIELD) or "")[:80].replace("\n", " ")
        _ok(f"{name}: search OK  (top-1 score={top.score:.4f}, preview='{text_preview}...')")
        return True
    except Exception as e:
        _fail(f"{name}: search failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Milvus sanity check for CR-MSR M1.5")
    ap.add_argument("--embed_model_path", required=True,
                    help="path to gte-large-en-v1.5 (or whatever embedder)")
    ap.add_argument("--milvus_host", default="localhost")
    ap.add_argument("--milvus_port", type=int, default=19530)
    args = ap.parse_args()

    print(f"=== Milvus sanity check ({args.milvus_host}:{args.milvus_port}) ===")
    fails = 0

    # 1) connection
    if not check_connection(args.milvus_host, args.milvus_port):
        sys.exit(1)

    # 2) collections
    cols = check_collections()
    fails += sum(1 for v in cols.values() if v is None)

    # 3) schema dims
    dims = []
    for name, col in cols.items():
        if col is None:
            continue
        d = check_schema(col, name)
        if d < 0:
            fails += 1
        else:
            dims.append((name, d))

    # 4) embedder
    embedder, embed_dim = check_embedder(args.embed_model_path)
    if embedder is None:
        fails += 1

    # 5) dim consistency check
    if embed_dim > 0 and dims:
        for name, d in dims:
            if d != embed_dim:
                _fail(f"{name}: dim mismatch — collection={d} vs embedder={embed_dim}")
                fails += 1
            else:
                _ok(f"{name}: dim match ({d})")

    # 6) load + smoke search
    if embedder is not None:
        for name, col in cols.items():
            if col is None:
                continue
            if not load_collection(col, name):
                fails += 1
                continue
            if not smoke_search(col, embedder, name):
                fails += 1

    # Summary
    print()
    if fails == 0:
        print("=== ALL CHECKS PASSED ===")
        sys.exit(0)
    else:
        print(f"=== {fails} CHECK(S) FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
