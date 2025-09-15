#!/usr/bin/env python3
"""
Compute in-group ranking metrics between an original proxy_score.csv (ground truth)
and a perturbed one (test):

  • nDCG  (graded, top-weighted)
  • PNR   (Pairwise Non-Reversal rate = 1 - normalized inversions)
  • Spearman's footrule distance (normalized) and similarity (= 1 - distance)

Each CSV must contain:
  - id_ids
  - idx_in_id_group
  - rewards

For each id_ids, we intersect rows by (id_ids, idx_in_id_group) and:
  - Ground-truth order  : sort by rewards_orig (desc, stable)
  - Test order          : sort by rewards_test (desc, stable)
  - nDCG                : relevance = n, n-1, ..., 1 from ground-truth ranks
  - PNR                 : fraction of pairs whose relative order matches GT
  - Footrule (normalized):
        F = sum_i |r_gt(i) - r_test(i)|, using 1-based ranks within the intersection
        F_max = floor(n^2 / 2)
        F_norm = F / F_max, similarity = 1 - F_norm

Macro = mean over groups (skip NaN); Micro pools denominators (IDCG, pair counts, F_max).

Usage:
  python rank_metrics_by_group.py \
    --original path/to/proxy_score_original.csv \
    --test path/to/proxy_score_perturbed.csv \
    --k 50 \
    --scheme exp \
    --out_csv per_group_metrics.csv
"""

import argparse
import math
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute in-group nDCG/PNR/Footrule metrics between original and perturbed proxy scores."
    )
    p.add_argument("--original", required=False, help="Path to original proxy_score.csv (ground truth).")
    p.add_argument("--test", required=False, help="Path to perturbed proxy_score.csv (test).")
    p.add_argument("--k", type=int, default=None, help="Cutoff K for nDCG@K. Default: full intersecting group.")
    p.add_argument("--scheme", choices=["exp", "linear"], default="exp",
                   help="nDCG gain scheme: 'exp' uses (2^rel - 1); 'linear' uses rel.")
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
        return [max(0.0, (2.0 ** float(r)) - 1.0) for r in rels]
    return [max(0.0, float(r)) for r in rels]


def _stable_mergesort_inversions(arr: List[int]) -> int:
    """Count inversions in arr via mergesort (O(n log n)), stable w.r.t ties."""
    def sort_count(a):
        n = len(a)
        if n <= 1: return a, 0
        mid = n // 2
        L, invL = sort_count(a[:mid])
        R, invR = sort_count(a[mid:])
        i = j = inv = 0
        merged = []
        inv += invL + invR
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:              # stable: ties keep order (no inversion)
                merged.append(L[i]); i += 1
            else:
                merged.append(R[j]); j += 1
                inv += (len(L) - i)
        merged.extend(L[i:]); merged.extend(R[j:])
        return merged, inv
    _, inv = sort_count(list(arr))
    return inv


def _prepare_group(df_orig_gid: pd.DataFrame, df_test_gid: pd.DataFrame) -> Tuple[pd.DataFrame, List[int], Dict[int, int]]:
    """
    Align and sort by ground-truth. Returns:
      merged: DataFrame intersection, sorted by rewards_orig desc (stable)
      test_rank_idx_list: idx_in_id_group in test ranking order desc (stable)
      gt_pos_by_idx: mapping idx_in_id_group -> ground-truth position (0 is best)
    """
    merged = pd.merge(
        df_orig_gid[["id_ids", "idx_in_id_group", "rewards"]],
        df_test_gid[["id_ids", "idx_in_id_group", "rewards"]],
        on=["id_ids", "idx_in_id_group"],
        suffixes=("_orig", "_test"),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) == 0:
        return merged, [], {}

    merged = merged.sort_values("rewards_orig", ascending=False, kind="mergesort").reset_index(drop=True)
    merged["gt_pos"] = merged.index  # 0 is best

    test_sorted = merged.sort_values("rewards_test", ascending=False, kind="mergesort")
    test_rank_idx_list = test_sorted["idx_in_id_group"].tolist()

    gt_pos_by_idx = dict(zip(merged["idx_in_id_group"].tolist(), merged["gt_pos"].tolist()))
    return merged, test_rank_idx_list, gt_pos_by_idx


def compute_group_ndcg(
    merged: pd.DataFrame, test_rank_idx_list: List[int], gt_pos_by_idx: Dict[int, int],
    k: int, scheme: str
) -> Tuple[float, int, int, float, float]:
    """Returns (ndcg, n_items, k_used, dcg_val, idcg_val)."""
    n = len(merged)
    if n == 0:
        return (float("nan"), 0, 0, 0.0, 0.0)

    rels_sorted = (n - merged["gt_pos"]).astype(float).tolist()  # n..1
    gains_ideal = gains_from_relevance(rels_sorted, scheme=scheme)
    k_used = n if (k is None or k > n) else k
    idcg_val = dcg(gains_ideal, k=k_used)
    if idcg_val <= 0:
        return (float("nan"), n, k_used, 0.0, 0.0)

    rel_by_idx = {idx: float(n - gt_pos) for idx, gt_pos in gt_pos_by_idx.items()}
    rel_in_test_order = [rel_by_idx[idx] for idx in test_rank_idx_list]
    gains_test = gains_from_relevance(rel_in_test_order, scheme=scheme)
    dcg_val = dcg(gains_test, k=k_used)

    return (dcg_val / idcg_val, n, k_used, dcg_val, idcg_val)


def compute_group_pnr(
    merged: pd.DataFrame, test_rank_idx_list: List[int], gt_pos_by_idx: Dict[int, int]
) -> Tuple[float, int, int, int]:
    """
    PNR = 1 - inversions / C(n,2), with respect to GT order.
    Returns: (pnr, n_pairs, inversions, concordant)
    """
    n = len(merged)
    if n <= 1:
        return (float("nan"), 0, 0, 0)

    perm = [gt_pos_by_idx[idx] for idx in test_rank_idx_list]  # GT positions in test order
    inversions = _stable_mergesort_inversions(perm)
    total_pairs = n * (n - 1) // 2
    concordant = total_pairs - inversions
    pnr = 1.0 - (inversions / total_pairs) if total_pairs > 0 else float("nan")
    return (pnr, total_pairs, inversions, concordant)


def compute_group_footrule(
    merged: pd.DataFrame, test_rank_idx_list: List[int], gt_pos_by_idx: Dict[int, int]
) -> Tuple[float, float, int, float]:
    """
    Spearman's footrule (normalized):
      F      = sum_i |r_gt(i) - r_test(i)|  using 1-based ranks
      F_max  = floor(n^2 / 2)
      F_norm = F / F_max
    Returns: (F_norm, F, F_max, n)
    """
    n = len(merged)
    if n <= 1:
        return (float("nan"), 0.0, 0, n)

    # 1-based GT ranks
    gt_rank_by_idx = {idx: pos + 1 for idx, pos in gt_pos_by_idx.items()}

    # 1-based TEST ranks
    test_rank_by_idx = {idx: rank for rank, idx in enumerate(test_rank_idx_list, start=1)}

    F = 0.0
    for idx in merged["idx_in_id_group"]:
        F += abs(gt_rank_by_idx[idx] - test_rank_by_idx[idx])

    F_max = (n * n) // 2
    if F_max == 0:
        return (float("nan"), F, F_max, n)

    return (F / F_max, F, F_max, n)


def main():
    args = parse_args()

    #args.original ="/mnt/nvme/jiabin/lora-boosting/lora-boosting/rlhf/bon/step3_obtain_proxy_score/gemma-2b-it/grm_0perturb/proxy_score.csv"
    #args.test ="/mnt/nvme/jiabin/lora-boosting/lora-boosting/rlhf/bon/step3_obtain_proxy_score/gemma-2b-it/grm_10perturb/proxy_score.csv"

    args.original ="/mnt/nvme/jiabin/lora-boosting/lora-boosting/rlhf/bon/step3_obtain_proxy_score/gemma-2b-it/avg_0perturb/proxy_score.csv"
    args.test ="/mnt/nvme/jiabin/lora-boosting/lora-boosting/rlhf/bon/step3_obtain_proxy_score/gemma-2b-it/avg_10perturb/proxy_score.csv"

    #args.original ="/mnt/nvme/jiabin/lora-boosting/lora-boosting/rlhf/bon/step3_obtain_proxy_score/gemma-2b-it/bt_0perturb/proxy_score.csv"
    #args.test ="/mnt/nvme/jiabin/lora-boosting/lora-boosting/rlhf/bon/step3_obtain_proxy_score/gemma-2b-it/bt_8perturb/proxy_score.csv"
    
    #args.original ="/mnt/nvme/jiabin/lora-boosting/lora-boosting/rlhf/bon/step3_obtain_proxy_score/gemma-2b-it/xgb3lora_0perturb/proxy_score.csv"
    #args.test ="/mnt/nvme/jiabin/lora-boosting/lora-boosting/rlhf/bon/step3_obtain_proxy_score/gemma-2b-it/xgb3lora_10perturb/proxy_score.csv"

    df_orig = pd.read_csv(args.original)
    df_test = pd.read_csv(args.test)
    _require_columns(df_orig, "original CSV")
    _require_columns(df_test, "test CSV")

    df_orig = df_orig.copy()
    df_test = df_test.copy()
    df_orig["idx_in_id_group"] = df_orig["idx_in_id_group"].astype(int)
    df_test["idx_in_id_group"] = df_test["idx_in_id_group"].astype(int)

    # De-duplicate on (id, idx)
    df_orig = df_orig.drop_duplicates(subset=["id_ids", "idx_in_id_group"], keep="first")
    df_test = df_test.drop_duplicates(subset=["id_ids", "idx_in_id_group"], keep="first")

    ids_in_both = sorted(set(df_orig["id_ids"]).intersection(set(df_test["id_ids"])))

    records = []

    # Micro accumulators
    sum_dcg = 0.0
    sum_idcg = 0.0
    concordant_total = 0
    pairs_total = 0
    footrule_F_total = 0.0
    footrule_Fmax_total = 0

    used_groups = 0

    for gid in ids_in_both:
        g_orig = df_orig[df_orig["id_ids"] == gid]
        g_test = df_test[df_test["id_ids"] == gid]

        merged, test_rank_idx_list, gt_pos_by_idx = _prepare_group(g_orig, g_test)
        n_inter = len(merged)

        if n_inter < args.min_group_size:
            records.append({
                "id_ids": gid,
                "n_items_intersection": n_inter,
                "k_used": 0,
                "ndcg": float("nan"),
                "pnr": float("nan"),
                "footrule_norm": float("nan"),
                "footrule_similarity": float("nan"),
                "footrule_F": 0.0,
                "footrule_Fmax": 0,
                "n_pairs": 0
            })
            continue

        nd, _, k_used, dcg_val, idcg_val = compute_group_ndcg(merged, test_rank_idx_list, gt_pos_by_idx, args.k, args.scheme)
        pnr, n_pairs, inversions, concordant = compute_group_pnr(merged, test_rank_idx_list, gt_pos_by_idx)
        f_norm, F, F_max, _ = compute_group_footrule(merged, test_rank_idx_list, gt_pos_by_idx)

        # Micro sums
        if not math.isnan(nd):
            sum_dcg += dcg_val
            sum_idcg += idcg_val
        if n_pairs > 0:
            concordant_total += concordant
            pairs_total += n_pairs
        if not math.isnan(f_norm):
            footrule_F_total += F
            footrule_Fmax_total += F_max

        used_groups += 1
        records.append({
            "id_ids": gid,
            "n_items_intersection": n_inter,
            "k_used": k_used,
            "ndcg": nd,
            "pnr": pnr,
            "footrule_norm": f_norm,
            "footrule_similarity": (1.0 - f_norm) if not math.isnan(f_norm) else float("nan"),
            "footrule_F": F,
            "footrule_Fmax": F_max,
            "n_pairs": n_pairs
        })

    df_out = pd.DataFrame(records)

    macro_ndcg = df_out["ndcg"].mean(skipna=True) if not df_out.empty else float("nan")
    macro_pnr  = df_out["pnr"].mean(skipna=True) if not df_out.empty else float("nan")
    macro_fnr  = df_out["footrule_norm"].mean(skipna=True) if not df_out.empty else float("nan")
    macro_fsim = df_out["footrule_similarity"].mean(skipna=True) if not df_out.empty else float("nan")

    micro_ndcg = (sum_dcg / sum_idcg) if sum_idcg > 0 else float("nan")
    micro_pnr  = (concordant_total / pairs_total) if pairs_total > 0 else float("nan")
    micro_fnr  = (footrule_F_total / footrule_Fmax_total) if footrule_Fmax_total > 0 else float("nan")
    micro_fsim = (1.0 - micro_fnr) if not math.isnan(micro_fnr) else float("nan")

    print(f"Groups in both: {len(ids_in_both)}; used (>= {args.min_group_size} items): {used_groups}")
    print(f"Macro nDCG: {macro_ndcg:.6f}" if not math.isnan(macro_ndcg) else "Macro nDCG: NaN")
    #print(f"Micro nDCG: {micro_ndcg:.6f}" if not math.isnan(micro_ndcg) else "Micro nDCG: NaN")
    print(f"Macro PNR:  {macro_pnr:.6f}"  if not math.isnan(macro_pnr)  else "Macro PNR: NaN")
    #print(f"Micro PNR:  {micro_pnr:.6f}"  if not math.isnan(micro_pnr)  else "Micro PNR: NaN")
    print(f"Macro Footrule (norm): {macro_fnr:.6f}" if not math.isnan(macro_fnr) else "Macro Footrule (norm): NaN")
    print(f"Macro Footrule similarity: {macro_fsim:.6f}" if not math.isnan(macro_fsim) else "Macro Footrule similarity: NaN")
    # print(f"Micro Footrule (norm): {micro_fnr:.6f}" if not math.isnan(micro_fnr) else "Micro Footrule (norm): NaN")
    # print(f"Micro Footrule similarity: {micro_fsim:.6f}" if not math.isnan(micro_fsim) else "Micro Footrule similarity: NaN")

    if args.out_csv:
        df_out.to_csv(args.out_csv, index=False)
        print(f"Per-group metrics saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
