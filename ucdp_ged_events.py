#!/usr/bin/env python3
"""
ucdp_ged_events.py
==================

Load and preprocess the UCDP Georeferenced Event Dataset (GED) into a clean
event representation for *multivariate temporal point-process* modeling -- i.e.
for studying self-excitation and episodic coordination across conflict streams.

UCDP GED is expert hand-coded organized-violence events (>=1 death), CC BY 4.0,
so ML use is fine; cite the dataset (see the codebook on https://ucdp.uu.se).
Current global version: 26.1; recent months live in the Candidate set (26.0.X).

Why this dataset suits an identifiability/recovery paper: each event carries
date_start, date_end and a date_prec flag (1 = exact day; higher = coarser, with
[date_start, date_end] bracketing the uncertainty). Temporal imprecision is thus
LABELLED rather than hidden, so you can (a) work on a clean exact-day subset and
(b) study how identifiability/recovery degrade as coarser events are admitted.

Two ways to get the data:
  1. LOCAL FILE (simplest, fully reproducible): download GED from
     https://ucdp.uu.se/downloads/ (accept CC BY terms), then point --ged-csv
     at the file. Works with the CSV or the Excel export.
  2. API (programmatic / recent Candidate data): the production API now expects
     an access token. Register, then pass --token or set UCDP_TOKEN. Endpoint:
        https://ucdpapi.pcr.uu.se/api/gedevents/<version>?pagesize=1000&page=N
     The response is paged; iterate to TotalPages.

Outputs (current directory, prefixed):
  <prefix>_events.csv          tidy event table, one row per event:
        event_time    float   days since the earliest date_start
        date_start    datetime
        date_end      datetime
        date_prec     int     1 = exact day; higher = coarser
        interval_days int      date_end - date_start, in days (0 = exact)
        stream        the mark/dimension for the multivariate model
        dyad / country / region / type_of_violence / best (fatalities) / lat / lon
  <prefix>_events.parquet      same table, if pyarrow is installed
  <prefix>_streams.npz         {stream_id: sorted np.array of event_times}
  <prefix>_streams_index.json  {stream_id: event_count}

Run:
  python ucdp_ged_events.py --ged-csv GEDEvent_v26_1.csv \
      --stream-by dyad --violence-types 1 --exact-dates-only

  UCDP_TOKEN=xxxx python ucdp_ged_events.py --version 26.1 --stream-by country
"""

import os
import io
import csv
import json
import argparse

import numpy as np
import pandas as pd
import requests

API_BASE = "https://ucdpapi.pcr.uu.se/api"
RESOURCE = "gedevents"

# GED columns we keep (names are stable across versions; CSV == API JSON keys)
KEEP = ["id", "date_start", "date_end", "date_prec", "type_of_violence",
        "dyad_new_id", "dyad_name", "conflict_name", "side_a", "side_b",
        "country", "region", "latitude", "longitude", "best"]

STREAM_COLS = {
    "dyad": "dyad_name",
    "country": "country",
    "region": "region",
    "type_of_violence": "type_of_violence",
}


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def load_ged_local(path):
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"  loaded {len(df):,} rows from {path}")
    return df


def fetch_ged_api(version, token, pagesize=1000, extra_params=None):
    """Page through the UCDP GED API to TotalPages. Requires a token on the
    current production API."""
    headers = {"x-ucdp-access-token": token} if token else {}
    url = f"{API_BASE}/{RESOURCE}/{version}"
    rows, page, total_pages = [], 0, 1
    while page < total_pages:
        params = {"pagesize": pagesize, "page": page}
        if extra_params:
            params.update(extra_params)
        resp = requests.get(url, params=params, headers=headers, timeout=180)
        if resp.status_code in (401, 403):
            raise PermissionError(
                "UCDP API rejected the request (auth). Register for an access "
                "token and pass --token / set UCDP_TOKEN, or use --ged-csv with "
                "a file from https://ucdp.uu.se/downloads/.")
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("Result")
        if batch is None:  # defensive: find the first list-valued field
            batch = next((v for v in data.values() if isinstance(v, list)), [])
        total_pages = int(data.get("TotalPages", 1) or 1)
        rows.extend(batch)
        page += 1
        print(f"  API: page {page}/{total_pages}, {len(rows):,} events so far")
    df = pd.DataFrame.from_records(rows)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #
def process_ged(df, stream_by="dyad", violence_types=None, exact_dates_only=False):
    if stream_by not in STREAM_COLS:
        raise ValueError(f"stream_by must be one of {list(STREAM_COLS)}")
    if df.empty:
        raise ValueError("no rows -- check the file / filters")

    df = df.copy()
    keep = [c for c in KEEP if c in df.columns]
    df = df[keep]

    df["date_start"] = pd.to_datetime(df["date_start"], format="mixed", errors="coerce")
    df["date_end"] = pd.to_datetime(df["date_end"], format="mixed", errors="coerce")
    for c in ("latitude", "longitude", "best", "date_prec", "type_of_violence"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    n0 = len(df)
    df = df.dropna(subset=["date_start", "date_end"])

    if violence_types:
        df = df[df["type_of_violence"].isin([int(v) for v in violence_types])]
    if exact_dates_only:
        df = df[df["date_prec"] == 1]

    stream_col = STREAM_COLS[stream_by]
    df["stream"] = df[stream_col].astype("string").str.strip()
    df = df.dropna(subset=["stream"])
    df = df[(df["stream"] != "") & (df["stream"].str.lower() != "nan")]

    # continuous time axis (days since earliest event), plus interval width
    df = df.sort_values("date_start", kind="mergesort").reset_index(drop=True)
    t0 = df["date_start"].iloc[0]
    df["event_time"] = (df["date_start"] - t0).dt.total_seconds() / 86400.0
    df["interval_days"] = (df["date_end"] - df["date_start"]).dt.days

    print(f"  processed {len(df):,}/{n0:,} usable events across "
          f"{df['stream'].nunique()} streams; "
          f"span {df['event_time'].max():.0f} days")

    cols = ["event_time", "date_start", "date_end", "date_prec", "interval_days",
            "stream", "dyad_name", "conflict_name", "side_a", "side_b",
            "country", "region", "type_of_violence", "best",
            "latitude", "longitude"]
    return df[[c for c in cols if c in df.columns]]


def date_precision_report(df):
    """The principled analogue of a timestamp-heaping check. GED labels its
    temporal uncertainty, so quantify it before modelling on a continuous axis:
    only date_prec == 1 events are true single-day points; the rest are
    interval-censored over [date_start, date_end]."""
    n = len(df)
    exact = (df["date_prec"] == 1).mean()
    dup = df["event_time"].duplicated().mean()
    print("Temporal precision (date_prec):")
    counts = df["date_prec"].value_counts().sort_index()
    for k, v in counts.items():
        print(f"  date_prec = {int(k)}: {v:>7,}  ({v / n:5.1%})")
    print(f"  exact single-day events:      {exact:5.1%}")
    print(f"  median interval width (days): {df['interval_days'].median():.0f}")
    print(f"  duplicated event_times:       {dup:5.1%}")
    print("  -> treat date_prec > 1 as interval-censored over [date_start,")
    print("     date_end]; see the GED codebook for the day-width per level.")


def to_multivariate(df, out_prefix):
    streams = {str(s): g["event_time"].to_numpy()
               for s, g in df.groupby("stream", sort=True)}
    np.savez_compressed(f"{out_prefix}_streams.npz", **streams)
    with open(f"{out_prefix}_streams_index.json", "w") as f:
        json.dump({k: int(v.size) for k, v in streams.items()}, f, indent=2)
    print(f"  wrote {len(streams)} streams to {out_prefix}_streams.npz")
    return streams


def save_table(df, out_prefix):
    df.to_csv(f"{out_prefix}_events.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"  wrote {out_prefix}_events.csv")
    try:
        df.to_parquet(f"{out_prefix}_events.parquet")
        print(f"  wrote {out_prefix}_events.parquet")
    except Exception as e:
        print(f"  (parquet skipped: {e})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ged-csv", default=None,
                   help="path to a local GED CSV/Excel from ucdp.uu.se/downloads")
    p.add_argument("--version", default="26.1", help="GED version for the API")
    p.add_argument("--token", default=os.getenv("UCDP_TOKEN"),
                   help="UCDP API access token (or set UCDP_TOKEN)")
    p.add_argument("--stream-by", choices=list(STREAM_COLS), default="dyad")
    p.add_argument("--violence-types", nargs="*", default=None,
                   help="1=state-based, 2=non-state, 3=one-sided")
    p.add_argument("--exact-dates-only", action="store_true",
                   help="keep only date_prec == 1 (true single-day) events")
    p.add_argument("--out-prefix", default="ucdp_ged")
    return p.parse_args()


def main():
    a = parse_args()
    if a.ged_csv:
        raw = load_ged_local(a.ged_csv)
    else:
        print(f"Fetching GED {a.version} from the UCDP API ...")
        raw = fetch_ged_api(a.version, a.token)

    events = process_ged(raw, stream_by=a.stream_by,
                         violence_types=a.violence_types,
                         exact_dates_only=a.exact_dates_only)
    date_precision_report(events)
    save_table(events, a.out_prefix)
    to_multivariate(events, a.out_prefix)
    print("Done.")


if __name__ == "__main__":
    main()
