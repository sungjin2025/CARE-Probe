# -*- coding: utf-8 -*-
"""_check_norms.py — confirm whether stored doc vectors are L2-normalized.

Diagnoses the PubMed/Wiki strength=1.0 saturation: if the document vectors were
NOT normalized at index-build time, IP search returns ‖d‖·cos (scale ≫ 1), which
the runner then clamps to 1.0 — destroying the strength signal.

Expected if hypothesis holds:
  MedRAG_PubMed      L2 norms ≈ 15~25   (UN-normalized)
  MedRAG_Wikipedia   L2 norms ≈ 15~25   (UN-normalized)
  Medical_Guidelines L2 norms ≈ 1.0     (normalized → real cosine)

Usage (server):
  python _check_norms.py --milvus_host localhost --milvus_port 19530
"""
import argparse
import numpy as np
from pymilvus import connections, Collection, DataType

COLLECTIONS = ["MedRAG_PubMed", "MedRAG_Wikipedia", "Medical_Guidelines"]

VEC_TYPES = {DataType.FLOAT_VECTOR}
for _t in ("FLOAT16_VECTOR", "BFLOAT16_VECTOR"):
    if hasattr(DataType, _t):
        VEC_TYPES.add(getattr(DataType, _t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milvus_host", default="localhost")
    ap.add_argument("--milvus_port", type=int, default=19530)
    ap.add_argument("--sample", type=int, default=5, help="vectors to sample per collection")
    args = ap.parse_args()

    connections.connect(alias="default", host=args.milvus_host, port=str(args.milvus_port))
    print(f"=== doc-vector L2 norm check ({args.milvus_host}:{args.milvus_port}) ===")

    for name in COLLECTIONS:
        try:
            c = Collection(name)
            c.load()
            pk_name = pk_type = vec_name = None
            for f in c.schema.fields:
                if getattr(f, "is_primary", False):
                    pk_name, pk_type = f.name, f.dtype
                if f.dtype in VEC_TYPES:
                    vec_name = f.name
            expr = f'{pk_name} != ""' if pk_type == DataType.VARCHAR else f"{pk_name} >= 0"
            rows = c.query(expr=expr, output_fields=[vec_name], limit=args.sample)
            norms = [round(float(np.linalg.norm(np.asarray(r[vec_name], dtype=np.float32))), 3)
                     for r in rows]
            mean_norm = round(float(np.mean(norms)), 3) if norms else float("nan")
            verdict = "NORMALIZED (cosine OK)" if 0.9 <= mean_norm <= 1.1 else "UN-normalized (IP != cosine)"
            print(f"  {name:20s} pk={pk_name} vec={vec_name}  mean‖d‖={mean_norm}  {norms}  -> {verdict}")
        except Exception as e:
            print(f"  {name:20s} [FAIL] {e}")


if __name__ == "__main__":
    main()
