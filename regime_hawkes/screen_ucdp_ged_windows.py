from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def _parse_years(s: str | None):
    if not s:
        return None
    parts = [int(x.strip()) for x in s.split(',') if x.strip()]
    return set(parts)


def main():
    p = argparse.ArgumentParser(description="Screen UCDP GED event tables for dense dyad windows suitable for Hawkes spike-in experiments.")
    p.add_argument("--events", required=True, help="*_events.csv produced by ucdp_ged_events.py")
    p.add_argument("--group-by", default="country", choices=["country", "region", "conflict_name", "all"], help="background grouping unit")
    p.add_argument("--window-years", type=int, default=5)
    p.add_argument("--step-years", type=int, default=1)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-events", type=int, default=1500)
    p.add_argument("--min-topk-events", type=int, default=800)
    p.add_argument("--min-streams", type=int, default=12)
    p.add_argument("--country", default=None, help="optional exact country filter")
    p.add_argument("--region", default=None, help="optional exact region filter")
    p.add_argument("--years", default=None, help="optional comma-separated allowed start years")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.events, low_memory=False)
    if df.empty:
        raise SystemExit("No events loaded.")
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    df = df.dropna(subset=["date_start", "stream"])
    if args.country and "country" in df.columns:
        df = df[df["country"].astype(str) == args.country]
    if args.region and "region" in df.columns:
        df = df[df["region"].astype(str) == args.region]
    allowed_years = _parse_years(args.years)

    df["year"] = df["date_start"].dt.year.astype(int)
    if allowed_years is not None:
        df = df[df["year"].isin(allowed_years)]
    if df.empty:
        raise SystemExit("No events after filters.")

    if args.group_by == "all":
        groups = [("ALL", df)]
    else:
        col = args.group_by
        if col not in df.columns:
            raise SystemExit(f"Column {col!r} is missing from {args.events}")
        groups = [(str(k), g.copy()) for k, g in df.groupby(col, sort=False) if str(k) and str(k).lower() != "nan"]

    rows = []
    for group_name, g in groups:
        min_y = int(g["year"].min())
        max_y = int(g["year"].max())
        for y0 in range(min_y, max_y - args.window_years + 2, args.step_years):
            y1 = y0 + args.window_years
            w = g[(g["year"] >= y0) & (g["year"] < y1)]
            if len(w) < args.min_events:
                continue
            stream_counts = w["stream"].astype(str).value_counts()
            if len(stream_counts) < args.min_streams:
                continue
            top = stream_counts.head(args.top_k)
            topk_events = int(top.sum())
            if topk_events < args.min_topk_events:
                continue
            exact_pct = float((w["date_prec"] == 1).mean()) if "date_prec" in w.columns else np.nan
            row = {
                "rank_score": topk_events,
                "group_by": args.group_by,
                "group": group_name,
                "window_start_year": int(y0),
                "window_end_year": int(y1 - 1),
                "window_start": f"{y0}-01-01",
                "window_end": f"{y1}-01-01",
                "total_events": int(len(w)),
                "topk_events": topk_events,
                "n_streams": int(len(stream_counts)),
                "topk_streams": int(len(top)),
                "median_topk_count": float(top.median()),
                "min_topk_count": int(top.min()),
                "max_topk_count": int(top.max()),
                "exact_pct": exact_pct,
                "top_streams_json": top.to_json(),
            }
            # Convenience metadata when grouping by country/region/conflict.
            for c in ["country", "region", "conflict_name"]:
                if c in w.columns:
                    vals = w[c].dropna().astype(str).value_counts().head(3)
                    row[f"top_{c}"] = "; ".join(vals.index.tolist())
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        print("No candidate windows met the thresholds. Try lower --min-events/--min-topk-events or larger --window-years.")
    else:
        out = out.sort_values(["rank_score", "total_events"], ascending=False).reset_index(drop=True)
        out.insert(0, "rank", np.arange(len(out)))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out):,} candidate windows to {out_path}")
    if not out.empty:
        cols = ["rank", "group", "window_start_year", "window_end_year", "total_events", "topk_events", "n_streams", "median_topk_count", "exact_pct"]
        print(out[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
