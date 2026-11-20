#!/usr/bin/env python3
"""
formate_screen_update.py -- column-level screen for treated-specific rising
nonsynonymous variants in formate metabolism genes.

Revision of formate_screen.py following collaborator review (R. Hu). Changes
from the previous version, all of which alter results:

  1. SPECIFICITY IS COUNTED IN COLUMNS, NOT SAMPLES. The old rule (">=2
     treated samples, 0 untreated") passed a variant seen twice in one
     reactor. A variant must now be present in >= --min-treated-columns of
     the 3 treated columns and in 0 untreated columns.
  2. RISING IS SCORED PER COLUMN. Each treated column's own five-point
     trajectory is scored; a variant must rise in >= --min-rising-columns of
     the 3. Replicate-averaged metrics are still reported for comparability
     with the published method, but no longer decide anything.
  3. ABSENCE IN UNTREATED IS CHECKED AGAINST THE UNFILTERED inStrain OUTPUT.
     filtered_SNVs.tsv already cuts at var/ref >= 0.15, so an allele sitting
     at 0.10 in untreated reads as absent. With --raw-snvs, every candidate
     position is re-checked in the untreated samples at the raw level and
     anything present sub-threshold is flagged and dropped by default.
  4. BASELINE (DAY 1) PRESENCE IS FLAGGED AND DROPPED BY DEFAULT. Day 1 is
     before any hypophosphite, so a variant already in the treated columns
     then is a starting difference between columns, not a response.
  5. THE MAIN TABLE USES THE ZERO-FILL-ROBUST RULE. Zero-filled replicate
     means are reported but are not the basis of any call. Uncovered
     positions are carried as missing, not as zeros, everywhere that a
     decision is made, and are drawn as gaps.
  6. --column-col IS THE PREFERRED INPUT and the script refuses to run the
     column-level rules on a guessed mapping.
  7. LINKED-SWEEP DIAGNOSTIC. A variant rising because the strain carrying it
     expanded is not a variant under selection at that site: everything on
     that haplotype rises together. The screen therefore scores ALL
     nonsynonymous variants in each genome as a background and reports, per
     genome, what fraction of them rise. A formate-gene hit in a genome where
     most variants are rising is a passenger on a genome-wide shift. A
     one-sided Fisher test asks whether formate genes rise more often than the
     rest of their own genome.

No output states or implies that any variant is under selection. "Rising" is
a description of a frequency trajectory and nothing more.

Dependencies: pandas, numpy, matplotlib. scipy is NOT required.

Typical run:

  python3 formate_screen_update.py \
      --column-col Column \
      --gene-info  /path/to/gene_info_all_samples.tsv \
      --raw-snvs   /path/to/raw_inStrain_SNVs_all_samples.tsv \
      --outdir     ./formate_screen_update_out
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import sys
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BASE = ("/global/scratch/users/mattweaver/ColumnSeqData/"
        "15.2.inStrain_rerun/VariantCalling")
DRAM_DEFAULT = ("/global/scratch/users/mattweaver/ColumnSeqData/"
                "10.2.DRAMannotations/annotations.tsv")

DEF_SNVS = os.path.join(BASE, "results", "filtered_SNVs.tsv")
DEF_CLUSTERS = os.path.join(BASE, "results", "clustering_final",
                            "cluster_assignments.tsv")
DEF_GRANTHAM = os.path.join(BASE, "results", "dnds_grantham_split",
                            "grantham_per_variant_by_treatment.tsv")

TIMEPOINTS = [1, 11, 27, 50, 59]
BASELINE_DAY = 1
DOSE_SCHEDULE = {1: 0.0, 11: 0.1, 27: 0.1, 50: 1.0, 59: 10.0}

SELECTED_SIX = [
    "t1_0_1_6_bin.20.permissive",   # Fermenter, Bacteroides
    "t1_0_1_2_bin.40.orig",         # Fermenter, Eubacterium_O
    "t1_0_1_6_bin.40.strict",       # Methanogen, Methanothrix
    "t1_0_1_6_bin.26.permissive",   # Methanogen, Methanocorpusculum
    "t6_1_10_3_bin.66.orig",        # Syntroph, JAQUBU01
    "t6_1_10_3_bin.6.orig",         # Syntroph, UBA5314
]
COVERAGE_EXCLUDED = ["t6_1_10_3_bin.66.orig", "t6_1_10_3_bin.6.orig"]

# ---------------------------------------------------------------------------
# Formate vocabulary (unchanged from formate_screen.py)
# ---------------------------------------------------------------------------

FORMATE_KOS = {
    "K00656",                                                   # PFL
    "K00122", "K00123", "K00124", "K00125", "K00126", "K00127",  # FDH
    "K22515", "K22516", "K05299", "K15022", "K22340",
    "K21993", "K06212", "K21990",                               # transporters
    "K08177",                                                   # antiporter
    "K15832", "K15834", "K15836",                               # FHL
    "K22015",                                                   # HDCR
}

TEXT_INCLUDE = re.compile(r"formate|formyl", re.I)

TEXT_EXCLUDE = [
    ("purine_formyltransferase", re.compile(
        r"phosphoribosyl\w*[\s-]*formyltransferase"
        r"|phosphoribosylglycinamide"
        r"|phosphoribosylaminoimidazole"
        r"|glycinamide\s*ribonucleotide"
        r"|\bGAR\b[\s-]*formyltransferase"
        r"|\bAICAR\b"
        r"|\bpur[NTHUJK]\b", re.I)),
    ("deformylase", re.compile(r"deformylase", re.I)),
    ("met_trna_formyltransferase", re.compile(
        r"(?:methionyl|\bmet)\w*[\s-]*tRNA[\s-]*formyltransferase"
        r"|\bfmt\b", re.I)),
    ("rnr_class_III", re.compile(
        r"ribonucleoside-?\s*triphosphate\s+reductase|\bnrdD\b", re.I)),
    ("formylmethanofuran_dehydrogenase", re.compile(
        r"formylmethanofuran|\bfwd[A-H]?\b|\bfmd[A-H]?\b", re.I)),
]

PFAM_FORMATE = {
    "PF00384": "Molybdopterin oxidoreductase",
    "PF01568": "Molybdopterin dinucleotide binding",
    "PF09163": "Formate dehydrogenase N transmembrane",
    "PF13247": "4Fe-4S dicluster",
}
PFAM_SPECIFIC = {"PF09163"}

UNIREF_EXTRA = re.compile(
    r"\bFdh[A-Z]?\b|hydrogen-?dependent\s+CO2\s+reductase|\bHDCR\b", re.I)

GRANTHAM_BANDS = [(0, 50, "conservative"), (51, 100, "moderate"),
                  (101, 150, "radical"), (151, 10_000, "very_radical")]

SNV_USECOLS_WANTED = [
    "sample", "id", "Genome", "gene", "mutation", "mutation_type",
    "frequency", "position_coverage", "Treatment_group", "RealTimepoint",
    "scaffold", "position", "var_base", "con_base", "ref_base", "Column",
    "column", "Reactor", "reactor",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Log:
    def __init__(self, path):
        self.path = path
        self.lines = []

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.lines.append(str(msg))

    def section(self, title):
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def save(self):
        with open(self.path, "w") as fh:
            fh.write("\n".join(self.lines) + "\n")


def norm_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_col(df, candidates, label, required=True, log=None):
    lookup = {norm_name(c): c for c in df.columns}
    for cand in candidates:
        if norm_name(cand) in lookup:
            return lookup[norm_name(cand)]
    for cand in candidates:
        n = norm_name(cand)
        for k, v in lookup.items():
            if n and n in k:
                return v
    if required:
        raise SystemExit(
            f"\nERROR: could not find a column for '{label}'.\n"
            f"  Tried: {candidates}\n"
            f"  Available: {list(df.columns)}\n")
    if log:
        log(f"  NOTE: optional column '{label}' not found "
            f"(tried {candidates}); continuing without it.")
    return None


def read_tsv(path, label, log, usecols=None):
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: {label} not found at {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False,
                     usecols=usecols)
    log(f"  read {label}: {len(df):,} rows x {len(df.columns)} cols  ({path})")
    return df


def read_snvs_chunked(path, label, log, chunksize=2_000_000,
                      nonsyn_only=True):
    """Stream a large SNV table, keeping only the columns and rows needed.

    filtered_SNVs.tsv is ~2.2 GB; reading it whole as strings will not fit in
    a login-node session. This keeps ~10 columns and (by default) only
    nonsynonymous rows, and categorises the repeated string columns.
    """
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: {label} not found at {path}")
    header = pd.read_csv(path, sep="\t", nrows=0)
    usecols = [c for c in header.columns
               if norm_name(c) in {norm_name(x) for x in SNV_USECOLS_WANTED}]
    mt_col = next((c for c in header.columns
                   if norm_name(c) == "mutationtype"), None)
    log(f"  streaming {label} ({path})")
    log(f"    keeping columns: {usecols}")
    parts, n_in = [], 0
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, dtype=str,
                             chunksize=chunksize):
        n_in += len(chunk)
        if nonsyn_only and mt_col is not None:
            chunk = chunk[chunk[mt_col].astype(str).str.upper()
                          .str.startswith("N")]
        for c in chunk.columns:
            if norm_name(c) in ("sample", "genome", "treatmentgroup",
                                "mutationtype", "column", "reactor"):
                chunk[c] = chunk[c].astype("category")
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    log(f"    {n_in:,} rows read, {len(df):,} kept"
        f"{' (nonsynonymous only)' if nonsyn_only and mt_col else ''}")
    return df


def split_tokens(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val)
    if s.strip() in ("", "nan", "NA", "None"):
        return []
    return [t.strip() for t in re.split(r"[;,|]\s*", s) if t.strip()]


def excluded_reason(text):
    if not text:
        return None
    for name, rx in TEXT_EXCLUDE:
        if rx.search(text):
            return name
    return None


def spearman(x, y):
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    ok = (~x.isna()) & (~y.isna())
    if ok.sum() < 3:
        return np.nan
    xr, yr = x[ok].rank(), y[ok].rank()
    if xr.std(ddof=0) == 0 or yr.std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def fisher_greater(a, b, c, d):
    """One-sided Fisher exact (enrichment of a) without scipy.

    Table: [[a, b], [c, d]].
    """
    n = a + b + c + d
    if n == 0:
        return np.nan
    r1, c1 = a + b, a + c

    def lchoose(n_, k_):
        if k_ < 0 or k_ > n_:
            return -np.inf
        return (math.lgamma(n_ + 1) - math.lgamma(k_ + 1)
                - math.lgamma(n_ - k_ + 1))

    denom = lchoose(n, c1)
    lo, hi = max(0, c1 - (c + d)), min(r1, c1)
    p = 0.0
    for k in range(a, hi + 1):
        p += math.exp(lchoose(r1, k) + lchoose(n - r1, c1 - k) - denom)
    return float(min(1.0, max(0.0, p)))


def bh_adjust(pvals):
    p = np.asarray(pvals, dtype="float64")
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q
    idx = np.argsort(p[ok])
    ps = p[ok][idx]
    m = len(ps)
    adj = ps * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[idx] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


def grantham_band(d):
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "unknown"
    for lo, hi, name in GRANTHAM_BANDS:
        if lo <= d <= hi:
            return name
    return "unknown"


def parse_mutation(mut):
    if not mut:
        return (None, None, None)
    m = re.match(r"^\s*[NSIM]?\s*:?\s*([A-Z\*])\s*(\d+)\s*([A-Z\*])\s*$",
                 str(mut))
    if not m:
        m = re.search(r"([A-Z\*])(\d+)([A-Z\*])", str(mut))
    return (m.group(1), int(m.group(2)), m.group(3)) if m \
        else (None, None, None)


def parse_variant_id(vid):
    """'<scaffold>_<position>_<base>' -> (scaffold, position, base)."""
    m = re.match(r"^(.*)_(\d+)_([ACGTacgt])$", str(vid))
    if m:
        return (m.group(1), int(m.group(2)), m.group(3).upper())
    m = re.match(r"^(.*)_(\d+)$", str(vid))
    if m:
        return (m.group(1), int(m.group(2)), None)
    return (None, None, None)


# ---------------------------------------------------------------------------
# Gene set (unchanged logic)
# ---------------------------------------------------------------------------

def build_formate_gene_set(dram, log):
    gene_c = dram.columns[0]
    fasta_c = find_col(dram, ["fasta", "genome", "bin"], "genome/fasta",
                       False, log)
    ko_c = find_col(dram, ["ko_id", "kegg_id", "ko"], "ko_id")
    kegg_c = find_col(dram, ["kegg_hit", "kegg_description"], "kegg_hit")
    uniref_c = find_col(dram, ["uniref_hit", "uniref_description"],
                        "uniref_hit", False, log)
    pfam_c = find_col(dram, ["pfam_hits", "pfam"], "pfam_hits", False, log)
    log(f"  gene id column: '{gene_c}'; genome column: '{fasta_c}'")

    out = pd.DataFrame({
        "gene": dram[gene_c].astype(str),
        "genome": dram[fasta_c].astype(str) if fasta_c else "unknown",
    })
    ko_raw = dram[ko_c].fillna("")
    kegg_raw = dram[kegg_c].fillna("")
    uni_raw = (dram[uniref_c].fillna("") if uniref_c
               else pd.Series([""] * len(dram), index=dram.index))
    pfam_raw = (dram[pfam_c].fillna("") if pfam_c
                else pd.Series([""] * len(dram), index=dram.index))

    ko_hit, ko_which = [], []
    for v in ko_raw:
        hits = sorted({t.upper() for t in split_tokens(v)} & FORMATE_KOS)
        ko_hit.append(bool(hits))
        ko_which.append(";".join(hits))

    combined = (kegg_raw.astype(str) + " | " + uni_raw.astype(str)
                + " | " + pfam_raw.astype(str))
    excl_reason = [excluded_reason(s) or "" for s in combined]
    kegg_hit = [bool(TEXT_INCLUDE.search(str(v))) for v in kegg_raw]

    pfam_which, pfam_n, pfam_hit = [], [], []
    for v in pfam_raw:
        found = sorted({pf for pf in PFAM_FORMATE if pf in str(v)})
        pfam_which.append(";".join(found))
        pfam_n.append(len(found))
        pfam_hit.append(bool(set(found) & PFAM_SPECIFIC) or len(found) >= 2)

    uni_hit = [bool(TEXT_INCLUDE.search(str(v)) or UNIREF_EXTRA.search(str(v)))
               for v in uni_raw]

    out["ev_ko"] = ko_hit
    out["ko_matched"] = ko_which
    out["ev_kegg"] = kegg_hit
    out["ev_pfam"] = pfam_hit
    out["pfam_matched"] = pfam_which
    out["n_formate_pfam"] = pfam_n
    out["ev_uniref"] = uni_hit
    out["excluded_reason"] = excl_reason
    out["ko_id"] = ko_raw.values
    out["kegg_hit"] = kegg_raw.values
    out["uniref_hit"] = uni_raw.values
    out["pfam_hits"] = pfam_raw.values
    out["ev_pfam_weak"] = (out["n_formate_pfam"] == 1) & (~out["ev_pfam"])

    def tier(r):
        if r["ev_ko"]:
            return "A_KO_confirmed"
        if r["ev_kegg"] or r["ev_uniref"]:
            return "B_text_inferred"
        if r["ev_pfam"]:
            return "C_domain_inferred"
        if r["ev_pfam_weak"]:
            return "D_weak_domain_only"
        return ""

    out["evidence_tier"] = out.apply(tier, axis=1)
    out["exclusion_overridden_by_ko"] = (out["excluded_reason"] != "") \
        & out["ev_ko"]
    out["excluded"] = (out["excluded_reason"] != "") & (~out["ev_ko"])

    cand = out[out["evidence_tier"] != ""].copy()
    gene_set = cand[~cand["excluded"]].copy()
    dropped = cand[cand["excluded"]].copy()

    log(f"  genes with >=1 formate evidence hit: {len(cand):,}")
    log(f"  excluded as known unrelated annotations: {len(dropped):,}")
    for r, n in dropped["excluded_reason"].value_counts().items():
        log(f"    {r}: {n:,}")
    if int(out["exclusion_overridden_by_ko"].sum()):
        log(f"  kept despite exclusion-matching text (formate KO assigned): "
            f"{int(out['exclusion_overridden_by_ko'].sum()):,}")
    log(f"  formate gene set: {len(gene_set):,}")
    for t, n in gene_set["evidence_tier"].value_counts().sort_index().items():
        log(f"    {t}: {n:,}")
    return gene_set, out, dropped


# ---------------------------------------------------------------------------
# Trajectory scoring
# ---------------------------------------------------------------------------

def rise_metrics(freqs, days, prefix, min_frac_pos):
    """Score one trajectory. NaN = not covered, never treated as zero.

    Steps are taken between consecutive OBSERVED timepoints, so a gap does not
    create a spurious fall.
    """
    v = np.asarray(freqs, dtype="float64")
    d = np.asarray(days, dtype="float64")
    ok = ~np.isnan(v)
    n_obs = int(ok.sum())
    rho = spearman(d, v)
    if n_obs >= 2:
        vv = v[ok]
        steps = np.diff(vv)
        n_pos = int((steps > 0).sum())
        n_steps = len(steps)
        frac_pos = n_pos / n_steps if n_steps else np.nan
        net = float(vv[-1] - vv[0])
        final = float(vv[-1])
        peak = float(np.nanmax(vv))
        first = float(vv[0])
    else:
        n_pos = n_steps = 0
        frac_pos = net = final = peak = first = np.nan
    return {
        f"{prefix}_rho": rho,
        f"{prefix}_n_pos_steps": n_pos,
        f"{prefix}_n_steps": n_steps,
        f"{prefix}_frac_pos_steps": frac_pos,
        f"{prefix}_first_freq": first,
        f"{prefix}_net_increase": net,
        f"{prefix}_final_freq": final,
        f"{prefix}_peak_freq": peak,
        f"{prefix}_n_timepoints_obs": n_obs,
    }


def per_column_matrix(df, cols, columns_in_group, variant_ids):
    """{column: DataFrame(variant x timepoint)} of observed frequencies."""
    sub = df[df[cols["id"]].isin(variant_ids)]
    out = {}
    for c in columns_in_group:
        s = sub[sub["_col"] == c]
        p = (s.groupby([cols["id"], cols["tp"]])["_freq"].mean()
             .unstack() if len(s) else pd.DataFrame())
        for t in TIMEPOINTS:
            if t not in p.columns:
                p[t] = np.nan
        out[c] = p[TIMEPOINTS] if len(p) else pd.DataFrame(
            columns=TIMEPOINTS)
    return out


def group_matrices(df, cols, columns_in_group, variant_ids):
    """Replicate-averaged matrices: zero-filled and observed-only."""
    sub = df[df[cols["id"]].isin(variant_ids)]
    if sub.empty:
        e = pd.DataFrame(columns=TIMEPOINTS)
        return e, e, e
    agg = (sub.groupby([cols["id"], cols["tp"]])
           .agg(sum_freq=("_freq", "sum"), n_obs=("_freq", "size"))
           .reset_index())
    denom = len(columns_in_group)
    agg["freq_zf"] = agg["sum_freq"] / denom
    agg["freq_obs"] = agg["sum_freq"] / agg["n_obs"]

    def piv(c):
        p = agg.pivot(index=cols["id"], columns=cols["tp"], values=c)
        for t in TIMEPOINTS:
            if t not in p.columns:
                p[t] = np.nan
        return p[TIMEPOINTS]

    return piv("freq_zf").fillna(0.0), piv("freq_obs"), \
        piv("n_obs").fillna(0).astype(int)


def score_variants(df, cols, group_cols, variant_ids, args, prefix_group):
    """Per-column and group-level scoring for one treatment group."""
    variant_ids = list(variant_ids)
    if not variant_ids:
        return pd.DataFrame()
    percol = per_column_matrix(df, cols, group_cols, variant_ids)
    zf, obs, reps = group_matrices(df, cols, group_cols, variant_ids)

    rows = []
    for vid in variant_ids:
        rec = {"variant_id": vid}
        n_rising_cols = 0
        n_present_cols = 0
        col_nets, col_finals = [], []
        for c in group_cols:
            m = percol.get(c)
            series = (m.loc[vid].to_numpy(dtype="float64")
                      if (m is not None and len(m) and vid in m.index)
                      else np.full(len(TIMEPOINTS), np.nan))
            met = rise_metrics(series, TIMEPOINTS, f"col{c}",
                               args.col_min_frac_pos)
            rec.update(met)
            for t, val in zip(TIMEPOINTS, series):
                rec[f"col{c}_d{t}"] = val
            present = met[f"col{c}_n_timepoints_obs"] > 0
            n_present_cols += int(present)
            rising = (
                met[f"col{c}_n_timepoints_obs"] >= args.col_min_timepoints
                and not np.isnan(met[f"col{c}_frac_pos_steps"])
                and met[f"col{c}_frac_pos_steps"] >= args.col_min_frac_pos
                and not np.isnan(met[f"col{c}_rho"])
                and met[f"col{c}_rho"] >= args.min_rho
                and met[f"col{c}_net_increase"] >= args.min_net_increase
                and met[f"col{c}_final_freq"] >= args.min_final_freq)
            rec[f"col{c}_rising"] = bool(rising)
            n_rising_cols += int(rising)
            if not np.isnan(met[f"col{c}_net_increase"]):
                col_nets.append(met[f"col{c}_net_increase"])
                col_finals.append(met[f"col{c}_final_freq"])
        rec["n_columns_present"] = n_present_cols
        rec["n_columns_rising"] = n_rising_cols
        rec["median_column_net_increase"] = (float(np.median(col_nets))
                                             if col_nets else np.nan)
        rec["median_column_final_freq"] = (float(np.median(col_finals))
                                           if col_finals else np.nan)

        zs = (zf.loc[vid].tolist() if vid in zf.index
              else [np.nan] * len(TIMEPOINTS))
        ob = (obs.loc[vid].tolist() if vid in obs.index
              else [np.nan] * len(TIMEPOINTS))
        rp = (reps.loc[vid].tolist() if vid in reps.index
              else [0] * len(TIMEPOINTS))
        rec.update(rise_metrics(zs, TIMEPOINTS, "zf", args.col_min_frac_pos))
        rec.update(rise_metrics(ob, TIMEPOINTS, "obs", args.col_min_frac_pos))
        for t, a, b, c_ in zip(TIMEPOINTS, zs, ob, rp):
            rec[f"freq_zf_d{t}"] = a
            rec[f"freq_obs_d{t}"] = b
            rec[f"n_cols_obs_d{t}"] = c_
        baseline_idx = TIMEPOINTS.index(BASELINE_DAY)
        rec["present_at_baseline"] = bool(rp[baseline_idx] > 0)
        rec["baseline_n_columns"] = int(rp[baseline_idx])
        rec["baseline_freq_obs"] = ob[baseline_idx]
        rows.append(rec)

    out = pd.DataFrame(rows)
    out["group"] = prefix_group
    # group-level rules, reported but not decisive
    out["rising_zf_published_rule"] = (
        (out["zf_frac_pos_steps"] >= args.col_min_frac_pos)
        & (out["zf_rho"] >= args.min_rho)
        & (out["zf_net_increase"] >= args.min_net_increase)
        & (out["zf_final_freq"] >= args.min_final_freq))
    out["rising_obs"] = (
        (out["obs_frac_pos_steps"] >= args.col_min_frac_pos)
        & (out["obs_rho"] >= args.min_rho)
        & (out["obs_net_increase"] >= args.min_net_increase)
        & (out["obs_final_freq"] >= args.min_final_freq))
    out["n_tp_observed"] = out[[f"n_cols_obs_d{t}"
                                for t in TIMEPOINTS]].gt(0).sum(axis=1)
    out["rise_robust_to_zerofill"] = (out["rising_zf_published_rule"]
                                      & out["rising_obs"]
                                      & (out["n_tp_observed"] >= 3))
    # THE decisive rule
    out["rising"] = ((out["n_columns_rising"] >= args.min_rising_columns)
                     & out["rise_robust_to_zerofill"])
    return out


def classify_specificity(df, cols, treated_cols, untreated_cols, args):
    """Column-level presence counts per variant."""
    t = df[df["_col"].isin(treated_cols)]
    u = df[df["_col"].isin(untreated_cols)]
    tc = t.groupby(cols["id"])["_col"].nunique().rename("n_columns_treated")
    uc = u.groupby(cols["id"])["_col"].nunique().rename(
        "n_columns_untreated")
    ts = t.groupby(cols["id"])[cols["sample"]].nunique().rename(
        "n_samples_treated")
    us = u.groupby(cols["id"])[cols["sample"]].nunique().rename(
        "n_samples_untreated")
    tf = t.groupby(cols["id"])["_freq"].max().rename("max_freq_treated")
    uf = u.groupby(cols["id"])["_freq"].max().rename("max_freq_untreated")
    tcov = t.groupby(cols["id"])["_cov"].mean().rename("cov_treated_at_pos")
    ucov = u.groupby(cols["id"])["_cov"].mean().rename(
        "cov_untreated_at_pos")

    base = (df.groupby(cols["id"])
            .agg(genome=(cols["genome"], "first"),
                 gene=(cols["gene"], "first"),
                 mutation=(cols["mutation"], "first")))
    var = (base.join([tc, uc, ts, us, tf, uf, tcov, ucov], how="left")
           .reset_index().rename(columns={cols["id"]: "variant_id"}))
    for c in ["n_columns_treated", "n_columns_untreated",
              "n_samples_treated", "n_samples_untreated"]:
        var[c] = var[c].fillna(0).astype(int)

    var["treated_specific"] = (
        (var["n_columns_treated"] >= args.min_treated_columns)
        & (var["n_columns_untreated"] == 0))
    var["untreated_specific"] = (
        (var["n_columns_untreated"] >= args.min_treated_columns)
        & (var["n_columns_treated"] == 0))
    var["shared"] = ((var["n_columns_treated"] > 0)
                     & (var["n_columns_untreated"] > 0))
    # the old sample-level rule, reported so the change is auditable
    var["treated_specific_OLD_sample_rule"] = (
        (var["n_samples_treated"] >= 2) & (var["n_samples_untreated"] == 0))
    return var


# ---------------------------------------------------------------------------
# Raw (unfiltered) untreated check
# ---------------------------------------------------------------------------

def check_raw_untreated(raw_path, candidates, untreated_samples, log,
                        chunksize=2_000_000):
    """Re-check candidate positions in the unfiltered inStrain SNV output.

    candidates: DataFrame with variant_id, scaffold, position, allele.
    Returns per-variant max untreated frequency of that allele at that
    position and how many untreated samples carried it at all.
    """
    if not os.path.exists(raw_path):
        raise SystemExit(f"ERROR: --raw-snvs not found at {raw_path}")
    want = set(zip(candidates["scaffold"].astype(str),
                   candidates["position"].astype("Int64").astype(str)))
    allele_of = dict(zip(zip(candidates["scaffold"].astype(str),
                             candidates["position"].astype("Int64")
                             .astype(str)),
                         candidates["allele"]))
    log(f"  re-checking {len(want):,} candidate positions in the unfiltered "
        f"inStrain output")
    header = pd.read_csv(raw_path, sep="\t", nrows=0)
    scaf_c = find_col(header, ["scaffold"], "scaffold (raw)")
    pos_c = find_col(header, ["position"], "position (raw)")
    samp_c = find_col(header, ["sample"], "sample (raw)")
    cov_c = find_col(header, ["position_coverage"], "position_coverage (raw)",
                     False, log)
    base_cols = {b: next((c for c in header.columns if c == b), None)
                 for b in ["A", "C", "T", "G"]}
    have_counts = all(base_cols.values())
    if have_counts:
        vb_c = vf_c = None
        log("    using A/C/T/G counts from the raw table")
    else:
        vb_c = find_col(header, ["var_base"], "var_base (raw)", False, log)
        vf_c = find_col(header, ["var_freq"], "var_freq (raw)", False, log)
        log("    no A/C/T/G counts; falling back to var_base + var_freq, "
            "which only sees the allele inStrain called as the variant")
    if not have_counts and not (vb_c and vf_c):
        raise SystemExit("ERROR: the raw SNV table needs either A/C/T/G "
                         "counts or var_base + var_freq.")

    keep = []
    for chunk in pd.read_csv(raw_path, sep="\t", dtype=str,
                             chunksize=chunksize):
        chunk = chunk[chunk[samp_c].isin(untreated_samples)]
        if chunk.empty:
            continue
        key = list(zip(chunk[scaf_c].astype(str), chunk[pos_c].astype(str)))
        mask = [k in want for k in key]
        chunk = chunk[mask]
        if chunk.empty:
            continue
        chunk["_key_scaf"] = chunk[scaf_c].astype(str)
        chunk["_key_pos"] = chunk[pos_c].astype(str)
        chunk["_allele"] = [allele_of.get((s, p)) for s, p
                            in zip(chunk["_key_scaf"], chunk["_key_pos"])]
        if have_counts:
            cnt = np.array([
                pd.to_numeric(chunk[base_cols[b]], errors="coerce")
                .fillna(0).to_numpy() for b in ["A", "C", "T", "G"]])
            order = {b: i for i, b in enumerate(["A", "C", "T", "G"])}
            idx = np.array([order.get(a, -1) for a in chunk["_allele"]])
            n_allele = np.where(idx >= 0,
                                cnt[np.clip(idx, 0, 3),
                                    np.arange(len(chunk))], np.nan)
            tot = (pd.to_numeric(chunk[cov_c], errors="coerce").to_numpy()
                   if cov_c else cnt.sum(axis=0))
            with np.errstate(invalid="ignore", divide="ignore"):
                chunk["_raw_freq"] = np.where(tot > 0, n_allele / tot, np.nan)
            chunk["_raw_count"] = n_allele
        else:
            vf = pd.to_numeric(chunk[vf_c], errors="coerce")
            chunk["_raw_freq"] = np.where(
                chunk[vb_c].astype(str) == chunk["_allele"], vf, 0.0)
            chunk["_raw_count"] = np.nan
        keep.append(chunk[["_key_scaf", "_key_pos", samp_c, "_raw_freq",
                           "_raw_count"]])
    if not keep:
        log("    no untreated rows matched; treating as no sub-threshold "
            "evidence (check that --raw-snvs covers the untreated samples)")
        return pd.DataFrame(columns=["scaffold", "position",
                                     "raw_untreated_max_freq",
                                     "raw_untreated_n_samples_with_allele"])
    raw = pd.concat(keep, ignore_index=True)
    raw["_present"] = raw["_raw_freq"].fillna(0) > 0
    g = (raw.groupby(["_key_scaf", "_key_pos"])
         .agg(raw_untreated_max_freq=("_raw_freq", "max"),
              raw_untreated_n_samples_with_allele=("_present", "sum"),
              raw_untreated_n_samples_covered=(samp_c, "nunique"))
         .reset_index()
         .rename(columns={"_key_scaf": "scaffold", "_key_pos": "position"}))
    log(f"    {len(g):,} candidate positions had untreated coverage in the "
        f"raw table")
    return g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_argparser(description):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=description)
    ap.add_argument("--snvs", default=DEF_SNVS)
    ap.add_argument("--dram", default=DRAM_DEFAULT)
    ap.add_argument("--clusters", default=DEF_CLUSTERS)
    ap.add_argument("--grantham", default=DEF_GRANTHAM)
    ap.add_argument("--gene-info", default=None,
                    help="concatenated inStrain gene_info across samples")
    ap.add_argument("--raw-snvs", default=None,
                    help="UNFILTERED inStrain SNV output across samples; used "
                         "to re-check candidate positions in untreated "
                         "columns below the 0.15 filter")
    ap.add_argument("--candidate-genomes", default=None)
    ap.add_argument("--outdir", default="./screen_out")
    ap.add_argument("--genome-scope", choices=["all", "candidates",
                                               "selected"], default="all")
    ap.add_argument("--include-weak-domain", action="store_true")
    # column-level rules
    ap.add_argument("--column-col", default=None,
                    help="column in the SNV table giving the reactor column. "
                         "STRONGLY preferred over --column-regex.")
    ap.add_argument("--column-regex", default=None,
                    help="fallback: regex with one capture group pulling the "
                         "reactor column out of the sample name")
    ap.add_argument("--treated-columns", default=None,
                    help="comma-separated treated column ids; default is "
                         "taken from Treatment_group")
    ap.add_argument("--min-treated-columns", type=int, default=2,
                    help="columns (not samples) a variant must appear in")
    ap.add_argument("--min-rising-columns", type=int, default=2,
                    help="columns the variant must be rising in")
    ap.add_argument("--col-min-timepoints", type=int, default=3,
                    help="observed timepoints needed to score one column")
    ap.add_argument("--col-min-frac-pos", type=float, default=0.75,
                    help="fraction of observed steps that must be positive")
    ap.add_argument("--min-rho", type=float, default=0.7)
    ap.add_argument("--min-net-increase", type=float, default=0.10)
    ap.add_argument("--min-final-freq", type=float, default=0.15)
    ap.add_argument("--min-gene-cov", type=float, default=5.0)
    # baseline / raw handling
    ap.add_argument("--keep-baseline-present", action="store_true",
                    help="keep variants already present in treated columns on "
                         "day 1 (flagged either way)")
    ap.add_argument("--keep-subthreshold-in-untreated", action="store_true",
                    help="keep variants found sub-threshold in untreated in "
                         "the raw table (flagged either way)")
    ap.add_argument("--no-permutation", action="store_true")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--max-facets", type=int, default=24,
                    help="cap on trajectory panels in the figure; with more "
                         "hits than this the strongest are plotted and the "
                         "rest are left to the tables")
    ap.add_argument("--min-background-variants", type=int, default=5,
                    help="genomes with fewer scored background variants are "
                         "left off the background figure (1 of 1 is not a "
                         "rate)")
    return ap


def resolve_columns(snv, cols, args, log):
    """Establish the reactor column for every row. Refuses to guess."""
    if args.column_col:
        c = find_col(snv, [args.column_col], "reactor column")
        snv["_col"] = snv[c].astype(str)
        log(f"  reactor column taken from '{c}' (--column-col)")
    elif args.column_regex:
        rx = re.compile(args.column_regex)
        snv["_col"] = snv[cols["sample"]].astype(str).map(
            lambda s: (rx.search(s).group(1) if rx.search(s) else None))
        log(f"  reactor column parsed from sample names with "
            f"{args.column_regex!r} -- VERIFY the mapping printed below")
    else:
        raise SystemExit(
            "\nERROR: no reactor column.\n"
            "  Every rule in this script counts columns, not samples, so the\n"
            "  column identity cannot be guessed. Pass --column-col <name>\n"
            "  (preferred), or --column-regex if the column is only encoded\n"
            "  in the sample name.\n"
            f"  Columns available: {list(snv.columns)}")

    found = sorted(x for x in snv["_col"].dropna().unique())
    log(f"  reactor columns found ({len(found)}): {found}")
    if snv["_col"].isna().any():
        bad = snv.loc[snv["_col"].isna(), cols["sample"]].unique()[:5]
        raise SystemExit(f"ERROR: reactor column missing for some rows, "
                         f"e.g. samples {list(bad)}")
    if len(found) != 6:
        log("  WARNING: expected 6 reactor columns.")

    mapping = (snv.groupby("_col", observed=True)[cols["group"]]
               .agg(lambda s: sorted(set(s))))
    log("  column -> treatment mapping:")
    for c, g in mapping.items():
        log(f"    {c}: {g}")
    mixed = [c for c, g in mapping.items() if len(g) > 1]
    if mixed:
        raise SystemExit(f"ERROR: columns {mixed} carry more than one "
                         "treatment label; the mapping is wrong.")

    if args.treated_columns:
        treated = [c.strip() for c in args.treated_columns.split(",")]
    else:
        treated = sorted(c for c, g in mapping.items() if g[0] == "Treated")
    untreated = sorted(c for c in found if c not in treated)
    log(f"  treated columns: {treated}")
    log(f"  untreated columns: {untreated}")
    if not treated or not untreated:
        raise SystemExit("ERROR: need at least one treated and one untreated "
                         "column.")
    return treated, untreated


def run(args, gene_mode, log_name, title):
    """gene_mode: 'formate' or 'all'."""
    os.makedirs(args.outdir, exist_ok=True)
    figdir = os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    log = Log(os.path.join(args.outdir, log_name))

    log(title)
    log(f"run started {datetime.now().isoformat(timespec='seconds')}")
    log(f"argv: {' '.join(sys.argv[1:])}")

    log.section("RULES IN FORCE")
    log(textwrap.dedent(f"""
    Specificity is counted in COLUMNS, not samples.
      A variant is treated-specific if it appears in >= {args.min_treated_columns}
      of the treated columns and in 0 untreated columns. The previous
      sample-level rule let one column observed at two timepoints pass; that
      rule is still reported as treated_specific_OLD_sample_rule so the
      change can be audited.

    Rising is scored PER COLUMN, then required in >= {args.min_rising_columns} columns.
      Each column's own five-point trajectory must have >= {args.col_min_timepoints}
      observed timepoints, >= {args.col_min_frac_pos:.0%} of its observed steps positive,
      Spearman rho >= {args.min_rho}, net increase >= {args.min_net_increase},
      final frequency >= {args.min_final_freq}. Replicate-averaged metrics are
      reported but decide nothing.

    Uncovered is missing, never zero.
      Steps are taken between consecutive OBSERVED timepoints, so a gap cannot
      manufacture a rise or a fall. The zero-filled replicate mean
      (rising_zf_published_rule) is retained only for comparability with the
      published method. The headline rule is
      rise_robust_to_zerofill AND rising in >= {args.min_rising_columns} columns.

    Day 1 is before any hypophosphite.
      A variant already present in the treated columns on day {BASELINE_DAY} is a
      starting difference between columns, not a response to dosing. Such
      variants are flagged present_at_baseline and
      {"KEPT (--keep-baseline-present)" if args.keep_baseline_present else "DROPPED from the headline table"}.

    Absence in untreated is checked against the unfiltered output.
      filtered_SNVs.tsv cuts at variant/reference ratio >= 0.15, so an allele
      at ~0.10 in untreated reads as absent. {"--raw-snvs supplied: every candidate position is re-checked." if args.raw_snvs else "--raw-snvs NOT supplied: every treated-specific call in this run is conditional on the 0.15 filter and should be treated as provisional."}

    Linked-sweep diagnostic.
      When a strain expands, every variant it carries rises together. The
      screen therefore scores ALL nonsynonymous variants per genome as a
      background and reports the genome's overall rising rate beside each hit.
      A one-sided Fisher test asks whether the gene set of interest rises more
      often than the rest of its own genome. A hit in a genome where much of
      the genome is rising is a passenger, not a site-specific change.

    Wording: nothing here is called positive selection. "Rising" describes a
      frequency trajectory. A rising variant is a candidate for follow-up.
    """).strip())

    # ---------------- gene set -----------------------------------------
    gene_set = screened_genes = dropped = None
    gene_to_tier = {}
    if gene_mode == "formate":
        log.section("STEP 0  Build the formate gene set")
        dram = read_tsv(args.dram, "DRAM annotations", log)
        gene_set, _, dropped = build_formate_gene_set(dram, log)
        tiers_in = ["A_KO_confirmed", "B_text_inferred", "C_domain_inferred"]
        if args.include_weak_domain:
            tiers_in.append("D_weak_domain_only")
        screened_genes = gene_set[gene_set["evidence_tier"]
                                  .isin(tiers_in)].copy()
        gene_to_tier = dict(zip(screened_genes["gene"],
                                screened_genes["evidence_tier"]))
        log(f"  tiers screened: {tiers_in}")
        log(f"  genes entering the screen: {len(screened_genes):,}")
        gene_set.to_csv(os.path.join(args.outdir, "formate_gene_set.tsv"),
                        sep="\t", index=False)
        dropped.to_csv(os.path.join(args.outdir,
                                    "formate_gene_set_excluded.tsv"),
                       sep="\t", index=False)
        log("  CAUTION: tier C rests on PF00384 / PF01568 / PF13247, shared "
            "with nitrate, DMSO and polysulfide reductases. Inspect by hand.")
    else:
        log.section("STEP 0  Gene annotation (all genes screened)")
        dram = read_tsv(args.dram, "DRAM annotations", log)
        gcol = dram.columns[0]
        fcol = find_col(dram, ["fasta", "genome", "bin"], "genome/fasta",
                        False, log)
        keep = [gcol] + [c for c in ["ko_id", "kegg_hit", "uniref_hit",
                                     "pfam_hits", "cazy_hits",
                                     "peptidase_hit"]
                         if c in dram.columns]
        gene_set = dram[keep].copy().rename(columns={gcol: "gene"})
        gene_set["genome"] = (dram[fcol].astype(str) if fcol else "unknown")
        screened_genes = gene_set
        log(f"  annotations carried for {len(gene_set):,} genes; no gene "
            "filter applied")

    # ---------------- SNVs ---------------------------------------------
    log.section("STEP 1  Load nonsynonymous variants")
    snv = read_snvs_chunked(args.snvs, "SNV table", log,
                            chunksize=args.chunksize, nonsyn_only=True)
    if snv.empty:
        log("No nonsynonymous rows. Stopping.")
        log.save()
        return

    cols = {
        "sample": find_col(snv, ["sample"], "sample"),
        "id": find_col(snv, ["id", "variant_id"], "variant id"),
        "genome": find_col(snv, ["Genome", "genome"], "Genome"),
        "gene": find_col(snv, ["gene"], "gene"),
        "mutation": find_col(snv, ["mutation"], "mutation"),
        "freq": find_col(snv, ["frequency", "var_freq"], "frequency"),
        "cov": find_col(snv, ["position_coverage", "coverage"],
                        "position_coverage"),
        "group": find_col(snv, ["Treatment_group", "treatment"],
                          "Treatment_group"),
        "tp": find_col(snv, ["RealTimepoint", "timepoint", "day"],
                       "RealTimepoint"),
    }
    snv["_freq"] = pd.to_numeric(snv[cols["freq"]], errors="coerce")
    snv["_cov"] = pd.to_numeric(snv[cols["cov"]], errors="coerce")
    snv[cols["tp"]] = pd.to_numeric(snv[cols["tp"]],
                                    errors="coerce").astype("Int64")

    def norm_group(v):
        s = str(v).strip().lower()
        return "Untreated" if (s.startswith("un") or "untreat" in s
                               or s in ("u", "ctrl", "control")) else "Treated"

    snv[cols["group"]] = snv[cols["group"]].astype(str).map(norm_group)
    log(f"  timepoints: "
        f"{sorted(snv[cols['tp']].dropna().unique().tolist())}")

    treated_cols, untreated_cols = resolve_columns(snv, cols, args, log)

    if args.genome_scope == "selected":
        scope = set(SELECTED_SIX)
    elif args.genome_scope == "candidates":
        if not args.candidate_genomes:
            raise SystemExit("--genome-scope candidates requires "
                             "--candidate-genomes")
        scope = {l.strip() for l in open(args.candidate_genomes) if l.strip()}
    else:
        scope = set(snv[cols["genome"]].astype(str).unique())
    snv = snv[snv[cols["genome"]].astype(str).isin(scope)]
    log(f"  genomes in scope: {len(scope):,}; rows: {len(snv):,}")

    # background = every nonsynonymous variant in scope
    background = snv.copy()
    if gene_mode == "formate":
        focus = snv[snv[cols["gene"]].astype(str).isin(gene_to_tier)].copy()
        log(f"  formate genes with >=1 nonsynonymous observation: "
            f"{focus[cols['gene']].nunique():,}")
        if focus.empty:
            log("  ERROR-LEVEL WARNING: no overlap between DRAM gene ids and "
                "SNV gene ids. Check the gene id format.")
            log.save()
            return
    else:
        focus = snv.copy()
    log(f"  focus rows: {len(focus):,}; background rows: {len(background):,}")

    # ---------------- Step 2: specificity -------------------------------
    log.section("STEP 2  Column-level specificity")
    var = classify_specificity(focus, cols, treated_cols, untreated_cols,
                               args)
    var["evidence_tier"] = var["gene"].map(gene_to_tier) if gene_to_tier \
        else pd.NA
    var["in_selected_six"] = var["genome"].isin(SELECTED_SIX)
    var["coverage_excluded_genome"] = var["genome"].isin(COVERAGE_EXCLUDED)

    if args.gene_info:
        gi = read_tsv(args.gene_info, "inStrain gene_info", log)
        gi_s = find_col(gi, ["sample"], "sample (gene_info)")
        gi_g = find_col(gi, ["gene"], "gene (gene_info)")
        gi_c = find_col(gi, ["coverage"], "coverage (gene_info)")
        gi["_cov"] = pd.to_numeric(gi[gi_c], errors="coerce")
        smap = (snv.drop_duplicates(cols["sample"])
                .set_index(cols["sample"])[cols["group"]].to_dict())
        gi["_grp"] = gi[gi_s].map(smap)
        tab = (gi[gi["_cov"] >= args.min_gene_cov]
               .groupby([gi_g, "_grp"])[gi_s].nunique().unstack(fill_value=0))
        for lbl, cname in (("Untreated", "n_untreated_samples_gene_callable"),
                           ("Treated", "n_treated_samples_gene_callable")):
            var[cname] = (var["gene"].map(tab[lbl]).fillna(0).astype(int)
                          if lbl in tab.columns else 0)
        log(f"  gene callability computed at >= {args.min_gene_cov}x")
    else:
        var["n_untreated_samples_gene_callable"] = pd.NA
        var["n_treated_samples_gene_callable"] = pd.NA
        log("  --gene-info not supplied: gene callability unknown.")

    log(f"  variants tested: {len(var):,}")
    log(f"    treated-specific (>= {args.min_treated_columns} treated "
        f"columns, 0 untreated): {int(var['treated_specific'].sum()):,}")
    log(f"    untreated-specific: {int(var['untreated_specific'].sum()):,}")
    log(f"    shared: {int(var['shared'].sum()):,}")
    log(f"    [old sample-level treated rule would have given: "
        f"{int(var['treated_specific_OLD_sample_rule'].sum()):,}]")
    var.to_csv(os.path.join(args.outdir, "step2_variant_specificity.tsv.gz"),
               sep="\t", index=False, compression="gzip")

    # ---------------- Step 3: rising ------------------------------------
    log.section("STEP 3  Per-column rise scoring")
    results = {}
    for label, flag, gcols_ in (("Treated", "treated_specific", treated_cols),
                                ("Untreated", "untreated_specific",
                                 untreated_cols)):
        ids = var.loc[var[flag], "variant_id"].tolist()
        log(f"  {label}: scoring {len(ids):,} {flag} variants across "
            f"{len(gcols_)} columns")
        sub = focus[focus["_col"].isin(gcols_)]
        sc = score_variants(sub, cols, gcols_, ids, args, label)
        results[label] = sc
        if sc.empty:
            continue
        log(f"    rising in >=1 column: "
            f"{int((sc['n_columns_rising'] >= 1).sum()):,}")
        log(f"    rising in >={args.min_rising_columns} columns: "
            f"{int((sc['n_columns_rising'] >= args.min_rising_columns).sum()):,}")
        log(f"    zero-fill-robust: "
            f"{int(sc['rise_robust_to_zerofill'].sum()):,}")
        log(f"    HEADLINE (both): {int(sc['rising'].sum()):,}")
        log(f"    [published zero-filled rule alone would give: "
            f"{int(sc['rising_zf_published_rule'].sum()):,}]")
        log(f"    present already on day {BASELINE_DAY}: "
            f"{int(sc['present_at_baseline'].sum()):,}")

    # ---------------- Step 3b: background rising rate per genome ---------
    log.section("STEP 3b  Genome-wide background (linked-sweep diagnostic)")
    bvar = classify_specificity(background, cols, treated_cols,
                                untreated_cols, args)
    bids = bvar.loc[bvar["treated_specific"], "variant_id"].tolist()
    log(f"  background treated-specific variants (all genes): {len(bids):,}")
    bsub = background[background["_col"].isin(treated_cols)]
    bsc = score_variants(bsub, cols, treated_cols, bids, args, "Treated")
    if not bsc.empty:
        bsc = bsc.merge(bvar[["variant_id", "genome", "gene"]],
                        on="variant_id", how="left")
        if not args.keep_baseline_present:
            bsc = bsc[~bsc["present_at_baseline"]]
        gbg = (bsc.groupby("genome")
               .agg(bg_variants_scored=("variant_id", "nunique"),
                    bg_variants_rising=("rising", "sum"))
               .reset_index())
        gbg["bg_frac_rising"] = (gbg["bg_variants_rising"]
                                 / gbg["bg_variants_scored"].replace(0,
                                                                     np.nan))
        # denominators, so the fraction is readable
        tot = (background.groupby(cols["genome"], observed=True)[cols["id"]]
               .nunique().rename("genome_nonsyn_variants_total"))
        gbg = gbg.merge(tot, left_on="genome", right_index=True, how="left")
        gbg.to_csv(os.path.join(args.outdir,
                                "step3b_genome_background.tsv"),
                   sep="\t", index=False)
        log("  per-genome background rising rate written to "
            "step3b_genome_background.tsv")
        top = gbg.sort_values("bg_frac_rising", ascending=False).head(10)
        log("")
        log(top.to_string(index=False))
    else:
        gbg = pd.DataFrame(columns=["genome", "bg_variants_scored",
                                    "bg_variants_rising", "bg_frac_rising"])
        log("  no background variants to score.")

    # ---------------- Step 4: characterise ------------------------------
    log.section("STEP 4  Characterise and filter the hits")

    gran = None
    if args.grantham and os.path.exists(args.grantham):
        gr = read_tsv(args.grantham, "Grantham per variant", log)
        gid = find_col(gr, ["id", "variant_id"], "variant id (grantham)")
        keep = {gid: "variant_id"}
        for k in ["aa_ref", "aa_alt", "codon", "grantham"]:
            c = find_col(gr, [k, f"{k}_distance", "codon_position"], k,
                         False, None)
            if c and c not in keep:
                keep[c] = k
        gran = gr[list(keep)].rename(columns=keep).drop_duplicates(
            "variant_id")
        if "grantham" in gran.columns:
            gran["grantham"] = pd.to_numeric(gran["grantham"],
                                             errors="coerce")
    else:
        log("  Grantham table not available; aa_ref/aa_alt/codon parsed from "
            "the mutation string, distance left as NA.")

    clus = None
    if args.clusters and os.path.exists(args.clusters):
        cl = read_tsv(args.clusters, "cluster assignments", log)
        cid = find_col(cl, ["id", "variant_id"], "variant id (clusters)")
        ccl = find_col(cl, ["cluster"], "cluster")
        clus = cl[[cid, ccl]].rename(columns={cid: "variant_id",
                                              ccl: "cluster"}) \
            .drop_duplicates("variant_id")

    gene_meta = (screened_genes.drop_duplicates("gene").set_index("gene")
                 if screened_genes is not None else pd.DataFrame())

    def characterise(sc):
        if sc.empty:
            return sc
        out = sc.merge(var, on="variant_id", how="left")
        if gran is not None:
            out = out.merge(gran, on="variant_id", how="left")
        if clus is not None:
            out = out.merge(clus, on="variant_id", how="left")
        out = out.merge(gbg, on="genome", how="left")
        parsed = out["mutation"].map(parse_mutation)
        for name, i in (("aa_ref", 0), ("codon", 1), ("aa_alt", 2)):
            vals = [p[i] for p in parsed]
            out[name] = out[name].fillna(pd.Series(vals, index=out.index)) \
                if name in out.columns else vals
        if "grantham" not in out.columns:
            out["grantham"] = np.nan
        out["grantham_band"] = out["grantham"].map(grantham_band)
        out["grantham_ge70"] = out["grantham"] >= 70
        out["grantham_gt100"] = out["grantham"] > 100
        for nm in ["ko_id", "kegg_hit", "uniref_hit", "pfam_hits",
                   "ko_matched", "pfam_matched"]:
            out[nm] = (out["gene"].map(gene_meta[nm])
                       if nm in getattr(gene_meta, "columns", []) else pd.NA)
        sid = out["variant_id"].map(parse_variant_id)
        out["scaffold"] = [s[0] for s in sid]
        out["position"] = pd.array([s[1] for s in sid], dtype="Int64")
        out["allele"] = [s[2] for s in sid]
        return out

    hits = {}
    for label in ("Treated", "Untreated"):
        sc = results.get(label, pd.DataFrame())
        full = characterise(sc)
        if full.empty:
            hits[label] = full
            continue
        full["dropped_reason"] = ""
        if not args.keep_baseline_present:
            full.loc[full["present_at_baseline"], "dropped_reason"] = \
                "present_at_baseline_day1"
        hits[label] = full

    # raw untreated re-check, treated candidates only
    th = hits.get("Treated", pd.DataFrame())
    if args.raw_snvs and len(th):
        log("")
        cand = th[th["rising"] & (th["dropped_reason"] == "")].copy()
        cand = cand.dropna(subset=["scaffold", "position", "allele"])
        if len(cand):
            untreated_samples = set(
                snv.loc[snv["_col"].isin(untreated_cols),
                        cols["sample"]].astype(str).unique())
            raw = check_raw_untreated(args.raw_snvs, cand,
                                      untreated_samples, log,
                                      chunksize=args.chunksize)
            if len(raw):
                raw["position"] = pd.to_numeric(raw["position"],
                                                errors="coerce").astype(
                                                    "Int64")
                th = th.merge(raw, on=["scaffold", "position"], how="left")
            else:
                for c in ["raw_untreated_max_freq",
                          "raw_untreated_n_samples_with_allele",
                          "raw_untreated_n_samples_covered"]:
                    th[c] = np.nan
            th["subthreshold_in_untreated"] = (
                th["raw_untreated_max_freq"].fillna(0) > 0)
            if not args.keep_subthreshold_in_untreated:
                th.loc[th["subthreshold_in_untreated"]
                       & (th["dropped_reason"] == ""),
                       "dropped_reason"] = "subthreshold_in_untreated_raw"
            log(f"  candidates found sub-threshold in untreated raw data: "
                f"{int(th['subthreshold_in_untreated'].sum()):,}")
            hits["Treated"] = th
    elif len(th):
        th["raw_untreated_max_freq"] = np.nan
        th["subthreshold_in_untreated"] = pd.NA
        hits["Treated"] = th

    front = (["variant_id", "genome", "in_selected_six",
              "coverage_excluded_genome", "gene", "evidence_tier",
              "ko_matched", "kegg_hit", "pfam_matched", "mutation",
              "aa_ref", "codon", "aa_alt", "grantham", "grantham_band",
              "n_columns_treated", "n_columns_untreated",
              "n_columns_present", "n_columns_rising",
              "median_column_net_increase", "median_column_final_freq"]
             + [f"col{c}_rising" for c in treated_cols]
             + [f"col{c}_d{t}" for c in treated_cols for t in TIMEPOINTS]
             + [f"freq_obs_d{t}" for t in TIMEPOINTS]
             + [f"freq_zf_d{t}" for t in TIMEPOINTS]
             + [f"n_cols_obs_d{t}" for t in TIMEPOINTS]
             + ["present_at_baseline", "baseline_n_columns",
                "baseline_freq_obs", "subthreshold_in_untreated",
                "raw_untreated_max_freq",
                "raw_untreated_n_samples_with_allele",
                "rising", "rise_robust_to_zerofill",
                "rising_zf_published_rule", "rising_obs",
                "treated_specific_OLD_sample_rule",
                "bg_variants_scored", "bg_variants_rising", "bg_frac_rising",
                "cov_treated_at_pos", "cov_untreated_at_pos",
                "n_untreated_samples_gene_callable",
                "n_treated_samples_gene_callable",
                "cluster", "dropped_reason"])

    for label, fname in (("Treated", "hits_treated_specific_rising.tsv"),
                         ("Untreated",
                          "hits_untreated_specific_rising.tsv")):
        full = hits.get(label, pd.DataFrame())
        if full.empty:
            pd.DataFrame(columns=front).to_csv(
                os.path.join(args.outdir, fname), sep="\t", index=False)
            log(f"  {label}: no variants.")
            continue
        full.to_csv(os.path.join(args.outdir,
                                 fname.replace(".tsv", "_all_scored.tsv.gz")),
                    sep="\t", index=False, compression="gzip")
        h = full[full["rising"] & (full["dropped_reason"] == "")
                 & (~full["coverage_excluded_genome"])].copy()
        ordered = [c for c in front if c in h.columns]
        h = h[ordered + [c for c in h.columns if c not in ordered]]
        sort_by = [c for c in ["genome", "gene", "n_columns_rising",
                               "median_column_net_increase"] if c in h.columns]
        h = h.sort_values(sort_by, ascending=[True, True, False, False][
            :len(sort_by)])
        h.to_csv(os.path.join(args.outdir, fname), sep="\t", index=False)
        hits[label] = full
        log(f"  {label}: {int(full['rising'].sum()):,} rising; "
            f"{len(h):,} in the headline table after dropping "
            f"baseline-present, sub-threshold-in-untreated and "
            f"coverage-excluded genomes")
        if "dropped_reason" in full.columns:
            for r, n in full.loc[full["rising"], "dropped_reason"] \
                    .value_counts().items():
                if r:
                    log(f"      dropped, {r}: {n:,}")

    # ---------------- Step 4b: enrichment vs genome background -----------
    if gene_mode == "formate" and len(gbg):
        log.section("STEP 4b  Do formate genes rise more than their genome?")
        th_all = hits.get("Treated", pd.DataFrame())
        rows = []
        if len(th_all):
            keep = th_all if args.keep_baseline_present else \
                th_all[~th_all["present_at_baseline"]]
            per = (keep.groupby("genome")
                   .agg(formate_scored=("variant_id", "nunique"),
                        formate_rising=("rising", "sum")).reset_index())
            per = per.merge(gbg, on="genome", how="left")
            for _, r in per.iterrows():
                a = int(r["formate_rising"])
                b = int(r["formate_scored"]) - a
                c = int(r.get("bg_variants_rising", 0) or 0) - a
                d = (int(r.get("bg_variants_scored", 0) or 0)
                     - int(r["formate_scored"]) - c)
                c, d = max(c, 0), max(d, 0)
                rows.append({
                    "genome": r["genome"],
                    "formate_scored": int(r["formate_scored"]),
                    "formate_rising": a,
                    "genome_scored": int(r.get("bg_variants_scored", 0) or 0),
                    "genome_rising": int(r.get("bg_variants_rising", 0) or 0),
                    "genome_frac_rising": r.get("bg_frac_rising", np.nan),
                    "p_fisher_greater": fisher_greater(a, b, c, d),
                })
        enr = pd.DataFrame(rows)
        if len(enr):
            enr["q_BH"] = bh_adjust(enr["p_fisher_greater"].tolist())
            enr = enr.sort_values("p_fisher_greater")
            enr.to_csv(os.path.join(args.outdir,
                                    "step4b_formate_vs_genome_background.tsv"),
                       sep="\t", index=False)
            log("")
            log(enr.head(20).to_string(index=False))
            log("  wrote step4b_formate_vs_genome_background.tsv")
            log("  A genome with a high genome_frac_rising is one where many "
                "variants are moving together, which is what a strain "
                "replacement looks like. Formate hits in such a genome are "
                "not site-specific findings.")
        else:
            log("  nothing to test.")

    # ---------------- figure --------------------------------------------
    th = hits.get("Treated", pd.DataFrame())
    plot_df = (th[th["rising"] & (th["dropped_reason"] == "")
                  & (~th["coverage_excluded_genome"])]
               if len(th) else pd.DataFrame())
    if len(plot_df):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # cap the grid: an all-genes run can have thousands of hit genes, and
        # a facet per gene would be unreadable as well as slow
        rank = (plot_df.groupby("gene")["median_column_net_increase"]
                .max().sort_values(ascending=False))
        genes = list(rank.index[:args.max_facets])
        n_all_genes = len(rank)
        if n_all_genes > len(genes):
            log(f"  figure capped at {args.max_facets} of {n_all_genes} hit "
                f"genes (strongest median per-column net increase); the full "
                f"set is in the tables")
        n = len(genes)
        ncol = min(3, max(1, n))
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(4.4 * ncol, 3.3 * nrow),
                                 squeeze=False, sharex=True, sharey=True)
        cmap = plt.get_cmap("tab10")
        for i, gene in enumerate(genes):
            ax = axes[i // ncol][i % ncol]
            sub = plot_df[plot_df["gene"] == gene]
            for _, r in sub.iterrows():
                for j, c in enumerate(treated_cols):
                    y = [r.get(f"col{c}_d{t}", np.nan) for t in TIMEPOINTS]
                    if all(pd.isna(v) for v in y):
                        continue
                    ax.plot(TIMEPOINTS, y, marker="o", ms=4, lw=1.3,
                            color=cmap(j % 10), alpha=.9,
                            label=f"col {c}" if i == 0 else None)
            for day, dose in DOSE_SCHEDULE.items():
                if dose == 0:
                    continue
                ax.axvline(day, color="0.7", lw=.7, ls="--", zorder=0)
                ax.text(day, 1.01, f"{dose:g}", fontsize=5, color="0.45",
                        ha="center", va="bottom")
            lbl = sub["mutation"].iloc[0] if len(sub) == 1 else \
                f"{len(sub)} variants"
            ax.set_title(f"{gene}\n{lbl}", fontsize=7)
            ax.set_ylim(0, 1)
            # always show the whole experiment: a variant absent on day 1 must
            # be visibly absent, not cropped out by autoscaling
            span = TIMEPOINTS[-1] - TIMEPOINTS[0]
            ax.set_xlim(TIMEPOINTS[0] - 0.05 * span,
                        TIMEPOINTS[-1] + 0.05 * span)
            ax.set_xticks(TIMEPOINTS)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=.25, lw=.5)
        for j in range(n, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.supxlabel("Day", fontsize=9)
        fig.supylabel("Variant frequency", fontsize=9)
        fig.suptitle(
            "Treated-specific rising nonsynonymous variants, one line per "
            "treated column\n"
            "gaps = position not covered (not plotted as zero); "
            "dashed verticals = hypophosphite dose (mM)", fontsize=9,
            y=0.995)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            # inside the first panel, so it cannot collide with the axis
            # label however many empty facets the grid ends up with
            axes[0][0].legend(handles, labels, fontsize=6, frameon=False,
                              loc="upper left", ncol=len(labels),
                              columnspacing=1.0, handlelength=1.2)
        fig.tight_layout(rect=(0, 0, 1, 0.97 if nrow > 1 else 0.90))
        p = os.path.join(figdir, "trajectories_by_column.png")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log(f"  wrote {p}")
    else:
        log("  no headline hits; no figure written.")

    # ---- the linked-sweep picture -------------------------------------
    if len(gbg):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        g = gbg[gbg["bg_variants_scored"] >= args.min_background_variants] \
            .sort_values("bg_frac_rising", ascending=True)
        n_off = len(gbg) - len(g)
        if len(g):
            fig, ax = plt.subplots(
                figsize=(7.2, max(2.4, 0.26 * len(g) + 1.2)))
            ypos = np.arange(len(g))
            ax.barh(ypos, g["bg_frac_rising"], color="#4C72B0", alpha=.85)
            ax.set_yticks(ypos)
            ax.set_yticklabels(g["genome"], fontsize=6)
            for y, (r, s) in enumerate(zip(g["bg_variants_rising"],
                                           g["bg_variants_scored"])):
                ax.text(min(g["bg_frac_rising"].iloc[y] + .012, .98), y,
                        f"{int(r)}/{int(s)}", va="center", fontsize=5.5,
                        color="0.3")
            ax.set_xlim(0, 1)
            ax.set_xlabel("fraction of the genome's treated-specific "
                          "nonsynonymous variants that are rising",
                          fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="x", alpha=.25, lw=.5)
            ax.set_title(
                "Linked-sweep diagnostic\n"
                "a genome near 1.0 is one where most variants move together, "
                "which is what a strain replacement looks like;\n"
                "a hit in such a genome is a passenger, not a "
                "site-specific change", fontsize=8)
            if n_off:
                ax.text(0.99, -0.06,
                        f"{n_off} genome(s) with fewer than "
                        f"{args.min_background_variants} scored variants "
                        f"omitted", transform=ax.transAxes, ha="right",
                        fontsize=6, color="0.4")
            fig.tight_layout()
            p2 = os.path.join(figdir, "genome_background_rising_rate.png")
            fig.savefig(p2, dpi=200, bbox_inches="tight")
            plt.close(fig)
            log(f"  wrote {p2}")
        else:
            log(f"  no genome had >= {args.min_background_variants} scored "
                "background variants; background figure not written.")

    # ---------------- Step 5: denominators + permutation -----------------
    log.section("STEP 5  Denominators and permutation null")
    rows = []
    for label, flag in (("Treated", "treated_specific"),
                        ("Untreated", "untreated_specific")):
        sc = results.get(label, pd.DataFrame())
        full = hits.get(label, pd.DataFrame())
        rows.append({
            "group": label,
            "genes_screened": (len(screened_genes)
                               if screened_genes is not None else 0),
            "genes_with_nonsyn_variants": focus[cols["gene"]].nunique(),
            "nonsyn_variants_tested": len(var),
            "group_specific_variants": int(var[flag].sum()),
            "rising_headline": int(sc["rising"].sum()) if len(sc) else 0,
            "rising_after_baseline_and_raw_filters":
                int((full["rising"] & (full["dropped_reason"] == "")).sum())
                if len(full) else 0,
            "rising_published_zf_rule":
                int(sc["rising_zf_published_rule"].sum()) if len(sc) else 0,
        })
    hr = pd.DataFrame(rows)
    hr.to_csv(os.path.join(args.outdir, "step5_hit_rates.tsv"), sep="\t",
              index=False)
    log("")
    log(hr.to_string(index=False))

    if not args.no_permutation:
        all_cols = sorted(set(treated_cols) | set(untreated_cols))
        true_t = set(treated_cols)
        null_rows = []
        for combo in itertools.combinations(all_cols, len(treated_cols)):
            combo = set(combo)
            kind = ("true_split" if combo in (true_t,
                                              set(all_cols) - true_t)
                    else "permuted")
            other = sorted(set(all_cols) - combo)
            v = classify_specificity(focus, cols, sorted(combo), other, args)
            ids = v.loc[v["treated_specific"], "variant_id"].tolist()
            sub = focus[focus["_col"].isin(combo)]
            sc = score_variants(sub, cols, sorted(combo), ids, args,
                                "pseudo")
            if len(sc) and not args.keep_baseline_present:
                sc = sc[~sc["present_at_baseline"]]
            null_rows.append({
                "pseudo_treated_columns": ",".join(sorted(combo)),
                "kind": kind,
                "n_group_specific": len(ids),
                "n_rising": int(sc["rising"].sum()) if len(sc) else 0})
        nullt = pd.DataFrame(null_rows).sort_values("n_rising",
                                                    ascending=False)
        nullt.to_csv(os.path.join(args.outdir,
                                  "step5_permutation_null.tsv"),
                     sep="\t", index=False)
        perm = nullt.loc[nullt["kind"] == "permuted", "n_rising"]
        true_n = nullt.loc[nullt["kind"] == "true_split", "n_rising"].max()
        log("")
        log(f"  permutation null over all C({len(all_cols)},"
            f"{len(treated_cols)}) splits:")
        log(f"    true split: {true_n} rising")
        if len(perm):
            log(f"    permuted (n={len(perm)}): median {perm.median():.1f}, "
                f"max {perm.max()}, splits >= true: "
                f"{int((perm >= true_n).sum())}")
            log("    Not a p-value: with six columns the permuted splits "
                "share columns with the true one.")

    log.section("DONE")
    log("Outputs in " + os.path.abspath(args.outdir))
    log("Nothing here is evidence of selection. A variant that is "
        "treated-specific and rising in two or three columns is a candidate "
        "for follow-up; check its genome's background rising rate "
        "(step3b/step4b) before reading it as site-specific.")
    log.save()


def main():
    ap = build_argparser(__doc__)
    ap.set_defaults(outdir="./formate_screen_update_out")
    args = ap.parse_args()
    run(args, gene_mode="formate", log_name="run_log.txt",
        title="formate_screen_update.py")


if __name__ == "__main__":
    main()
