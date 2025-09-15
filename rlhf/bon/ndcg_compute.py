#!/usr/bin/env python3
"""
Compute in-group nDCG between an original proxy_score.csv and a perturbed one.

Each CSV must contain:
  - id_ids              (group id)
  - idx_in_id_group     (per-id index, int)
  - rewards             (float)

For each id_ids:
  1) Ground truth relevance is derived from the original's ranking by rewards.
     Highest reward => highest relevance (n, n-1, ..., 1).
  2) Test ranking is the perturbed file's ranking by rewards.
  3) nDCG is computed per group, then summarized across groups.

Usage:
  python ndcg_by_group.py \
      --original path/to/proxy_score_original.csv \
      --test path/to/proxy_score_perturbed.csv \
      --k 50 \
      --scheme exp \
      --out_csv per_group_ndcg.csv
"""

import argparse
import math
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Compute in-group nDCG between original and perturbed proxy scores.")
    p.add_argument("--original", required=False, help="Path to original proxy_score.csv (ground truth).")
    p.add_argument("--test", required=False, help="Path to perturbed proxy_score.csv (test).")
    p.add_argument("--k", type=int, default=None, help="Cutoff K for nDCG@K. Default: use full group.")
    p.add_argument("--scheme", choices=["exp", "linear"], default="exp",
                   help="Gain scheme: 'exp' uses (2^rel - 1); 'linear' uses rel.")
    p.add_argument("--out_csv", default=None, help="Optional path to save per-group metrics as CSV.")
    p.add_argument("--min_group_size", type=int, default=2,
                   help="Skip groups with < this many intersecting items. Default: 2.")
    return p.parse_args()


def _require_columns(df: pd.DataFrame, name: str):
    needed = {"id_ids", "idx_in_id_group", "rewards"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def dcg(gains: List[float], k: int = None) -> float:
    if k is not None:
        gains = gains[:k]
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def gains_from_relevance(rels: List[float], scheme: str) -> List[float]:
    if scheme == "exp":
        # (2^rel - 1) with rel >= 0
        return [max(0.0, (2.0 ** r) - 1.0) for r in rels]
    # linear
    return [max(0.0, float(r)) for r in rels]


def compute_group_ndcg(
    df_orig_gid: pd.DataFrame,
    df_test_gid: pd.DataFrame,
    k: int,
    scheme: str
) -> Tuple[float, int, int, float, float]:
    """
    Returns: (ndcg, n_items_intersection, k_used, dcg_val, idcg_val)
    """
    # align on (id, idx)
    merged = pd.merge(
        df_orig_gid[["id_ids", "idx_in_id_group", "rewards"]],
        df_test_gid[["id_ids", "idx_in_id_group", "rewards"]],
        on=["id_ids", "idx_in_id_group"],
        suffixes=("_orig", "_test"),
        how="inner",
        validate="one_to_one",
    )

    n = len(merged)
    if n == 0:
        return (float("nan"), 0, 0, 0.0, 0.0)

    # Ground-truth ranking (by original rewards desc)
    merged = merged.sort_values("rewards_orig", ascending=False, kind="mergesort").reset_index(drop=True)

    # Relevance = rank-based, highest gets n, then n-1, ..., 1
    merged["rel"] = (n - merged.index).astype(float)

    # Ideal gains (IDCG): rels sorted descending already
    rels_sorted = merged["rel"].tolist()
    gains_ideal = gains_from_relevance(rels_sorted, scheme=scheme)
    k_used = n if (k is None or k > n) else k
    idcg_val = dcg(gains_ideal, k=k_used)
    if idcg_val <= 0:
        # If all relevance are zero (shouldn't happen here), return NaN
        return (float("nan"), n, k_used, 0.0, 0.0)

    # Build lookup rel by (idx)
    rel_by_idx: Dict[int, float] = dict(zip(merged["idx_in_id_group"].tolist(), merged["rel"].tolist()))

    # Test ranking by test rewards desc (only items in intersection; stable to preserve order on ties)
    test_rank = merged.sort_values("rewards_test", ascending=False, kind="mergesort")
    rel_in_test_order = [rel_by_idx[idx] for idx in test_rank["idx_in_id_group"].tolist()]
    gains_test = gains_from_relevance(rel_in_test_order, scheme=scheme)
    dcg_val = dcg(gains_test, k=k_used)

    return (dcg_val / idcg_val, n, k_used, dcg_val, idcg_val)


def main():
    args = parse_args()

    df_orig = pd.read_csv(args.original)
    df_test = pd.read_csv(args.test)

    _require_columns(df_orig, "original CSV")
    _require_columns(df_test, "test CSV")

    # Ensure consistent dtypes
    df_orig = df_orig.copy()
    df_test = df_test.copy()
    # cast idx to int (robustness)
    df_orig["idx_in_id_group"] = df_orig["idx_in_id_group"].astype(int)
    df_test["idx_in_id_group"] = df_test["idx_in_id_group"].astype(int)

    # Optional: drop exact duplicates on (id, idx) keeping first
    df_orig = df_orig.drop_duplicates(subset=["id_ids", "idx_in_id_group"], keep="first")
    df_test = df_test.drop_duplicates(subset=["id_ids", "idx_in_id_group"], keep="first")

    # Iterate groups present in either; we’ll compute only where both have members
    ids_in_both = sorted(set(df_orig["id_ids"]).intersection(set(df_test["id_ids"])))
    records = []
    sum_dcg = 0.0
    sum_idcg = 0.0
    used_groups = 0

    for gid in ids_in_both:
        g_orig = df_orig[df_orig["id_ids"] == gid]
        g_test = df_test[df_test["id_ids"] == gid]

        nd, n_inter, k_used, dcg_val, idcg_val = compute_group_ndcg(
            g_orig, g_test, k=args.k, scheme=args.scheme
        )
        if n_inter >= args.min_group_size and not math.isnan(nd):
            used_groups += 1
            sum_dcg += dcg_val
            sum_idcg += idcg_val
            records.append({
                "id_ids": gid,
                "n_items_intersection": n_inter,
                "k_used": k_used,
                "ndcg": nd
            })
        else:
            records.append({
                "id_ids": gid,
                "n_items_intersection": n_inter,
                "k_used": k_used,
                "ndcg": float("nan")
            })

    df_out = pd.DataFrame(records)

    # Macro: mean over groups (excluding NaN)
    macro_ndcg = df_out["ndcg"].mean(skipna=True) if not df_out.empty else float("nan")
    # Micro: sum DCG / sum IDCG
    micro_ndcg = (sum_dcg / sum_idcg) if sum_idcg > 0 else float("nan")

    print(f"Groups in both: {len(ids_in_both)}; used (>= {args.min_group_size} items): {used_groups}")
    print(f"Macro nDCG: {macro_ndcg:.6f}" if not math.isnan(macro_ndcg) else "Macro nDCG: NaN")
    print(f"Micro nDCG: {micro_ndcg:.6f}" if not math.isnan(micro_ndcg) else "Micro nDCG: NaN")

    if args.out_csv:
        df_out.to_csv(args.out_csv, index=False)
        print(f"Per-group metrics saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
