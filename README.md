# Identifiability and Recovery of Episodic Coordination in Switching Marked Point Processes
# Command pipeline to reproduce results published in the paper.

This README gives the commands used to regenerate the processed experiment files and reproduce the empirical results for:

**Identifiability and Recovery of Episodic Coordination in Switching Marked Point Processes**

The repository does **not** commit derived experiment data. The reviewer should place the raw/public data archives in the locations below, then run the commands in order. All processed event files, manifests, spike-ins, and results are generated locally.
For readers who want a conceptual map of the empirical design, this repository
also includes `ARCHITECTURE.md`. That file explains how the experiments map to
the paper's theorem-facing predictions: active exposure, weakest-link scaling,
posterior accuracy, and separability/collapse. It is intended as a guide to the
empirical method rather than as an additional result.
## 0. Repository layout

All reproduction scripts live at the repository root.

```text
regime-gated-hawkes/
  reviewer_upgrade_experiments.py
  reviewer_upgrade_crcns_hc_advisor_mp_SAFE.py
  reviewer_upgrade_design_ablations.py
  crypto_additive_spikein_compact.py
  make_5k_events.py
  ucdp_ged_events.py
  screen_ucdp_ged_windows.py

  regime_hawkes/
    __init__.py
    config.py
    simulate.py
    em.py
    estep.py
    mstep.py
    evaluate.py
    likelihood.py
    traces.py
    run_baseline_static_hawkes.py
    modular_hmm_hawkes_baseline.py
    run_baseline_spectral.py
    run_ucdp_ged_spikein_mp.py
    evaluate_ucdp_ged_spikein.py

  data/
    raw_crcns/
    processed_crcns/
    raw_crypto/
    processed_crypto/
      binance_vision/
    raw_ucdp/
    processed_ucdp/

  runs/
  paper_outputs/
```

## 1. Raw data placement

Download and place the two CRCNS hc-11 archives here:

```text
data/raw_crcns/Achilles_10252013.tar.gz
data/raw_crcns/Cicero_09012014.tar.gz
```

Download and place the Binance raw aggregate-trade files for the FTX window here:

```text
data/raw_crypto/binance_vision/
```

The crypto preprocessing command below expects Binance USD-M futures aggregate-trade CSV/ZIP files for the 20 symbols used in the paper and dates 2022-11-06 through 2022-11-11.

Download and place the UCDP GED export here, if reproducing the social-science experiments:

```text
data/raw_ucdp/GEDEvent_v26_1.csv
```

The UCDP scripts also support the GED API, but the paper reproduction path uses
the downloaded public CSV export for version 26.1.


The symbols are:

```text
FTTUSDT SOLUSDT BTCUSDT ETHUSDT BNBUSDT MATICUSDT AVAXUSDT DOGEUSDT ADAUSDT XRPUSDT LINKUSDT DOTUSDT LTCUSDT BCHUSDT ETCUSDT ATOMUSDT FILUSDT AAVEUSDT UNIUSDT NEARUSDT
```

## 2. Environment setup

Run from the repository root.

```powershell
cd regime-gated-hawkes

conda create -n hawkes-tmlr python=3.13 -y
conda activate hawkes-tmlr

python -m pip install --upgrade pip
pip install numpy pandas scipy matplotlib jax jaxlib tqdm requests openpyxl pyarrow
```

Set conservative thread limits:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:PYTHONUNBUFFERED = "1"
```

Create output directories:

```powershell
New-Item -ItemType Directory -Force -Path data/raw_crcns | Out-Null
New-Item -ItemType Directory -Force -Path data/processed_crcns | Out-Null
New-Item -ItemType Directory -Force -Path data/raw_crypto/binance_vision | Out-Null
New-Item -ItemType Directory -Force -Path data/processed_crypto/binance_vision | Out-Null
New-Item -ItemType Directory -Force -Path data/raw_ucdp | Out-Null
New-Item -ItemType Directory -Force -Path data/processed_ucdp | Out-Null
New-Item -ItemType Directory -Force -Path runs | Out-Null
New-Item -ItemType Directory -Force -Path paper_outputs | Out-Null
```

Smoke-test imports:

```powershell
python - <<'PY'
import numpy, pandas, scipy, matplotlib
import reviewer_upgrade_experiments
import reviewer_upgrade_crcns_hc_advisor_mp_SAFE
import reviewer_upgrade_design_ablations
import ucdp_ged_events
import screen_ucdp_ged_windows
from regime_hawkes.em import run_em
from regime_hawkes.estep import run_estep
print("imports ok")
PY
```

## 3. Unpack CRCNS archives

```powershell
New-Item -ItemType Directory -Force -Path data/raw_crcns/Achilles_10252013 | Out-Null
New-Item -ItemType Directory -Force -Path data/raw_crcns/Cicero_09012014 | Out-Null

tar -xzf data/raw_crcns/Achilles_10252013.tar.gz -C data/raw_crcns/Achilles_10252013
tar -xzf data/raw_crcns/Cicero_09012014.tar.gz -C data/raw_crcns/Cicero_09012014
```

## 4. Create CRCNS processed event files from unpacked archives

This command scans each unpacked CRCNS archive for spike-time/unit files and writes the standardized event format used by the experiments:

```text
time,actor,mark
```

The paper uses a single mark for CA1, so `mark=0` for every spike.

```powershell
@'
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat

RAW_ROOT = Path("data/raw_crcns")
OUT_ROOT = Path("data/processed_crcns")
SESSIONS = ["Achilles_10252013", "Cicero_09012014"]

def try_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    lower = {c.lower(): c for c in df.columns}
    if "time" in lower:
        time_col = lower["time"]
    elif "spike_time" in lower:
        time_col = lower["spike_time"]
    elif "spiketime" in lower:
        time_col = lower["spiketime"]
    elif "t" in lower:
        time_col = lower["t"]
    else:
        return None

    if "actor" in lower:
        actor_col = lower["actor"]
    elif "unit" in lower:
        actor_col = lower["unit"]
    elif "cell" in lower:
        actor_col = lower["cell"]
    elif "cluster" in lower:
        actor_col = lower["cluster"]
    else:
        actor_col = None

    out = pd.DataFrame()
    out["time"] = pd.to_numeric(df[time_col], errors="coerce")
    if actor_col is None:
        out["actor"] = 0
    else:
        out["actor"] = pd.to_numeric(df[actor_col], errors="coerce")

    out = out.dropna(subset=["time", "actor"])
    if out.empty:
        return None

    out["actor"] = out["actor"].astype(int)
    out["mark"] = 0
    return out[["time", "actor", "mark"]]

def flatten_numeric_object_array(x):
    vals = []
    arr = np.asarray(x)
    if arr.dtype == object:
        for item in arr.ravel():
            try:
                y = np.asarray(item, dtype=float).ravel()
                y = y[np.isfinite(y)]
                if y.size:
                    vals.append(y)
            except Exception:
                pass
    else:
        try:
            y = np.asarray(arr, dtype=float).ravel()
            y = y[np.isfinite(y)]
            if y.size:
                vals.append(y)
        except Exception:
            pass
    return vals

def try_mat(path):
    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception:
        return None

    candidate_keys = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        lk = k.lower()
        if any(tok in lk for tok in ["spike", "spk", "times", "res"]):
            candidate_keys.append(k)

    rows = []

    for k in candidate_keys:
        vals = flatten_numeric_object_array(mat[k])
        if len(vals) >= 2:
            for actor, times in enumerate(vals):
                for t in times:
                    rows.append((float(t), int(actor), 0))

    if not rows:
        return None

    out = pd.DataFrame(rows, columns=["time", "actor", "mark"])
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if out.empty:
        return None

    return out[["time", "actor", "mark"]]

def find_events(session_dir):
    # Prefer explicit processed-looking CSV files if the archive already contains them.
    csvs = list(session_dir.rglob("*.csv")) + list(session_dir.rglob("*.txt"))
    for p in csvs:
        df = try_csv(p)
        if df is not None and len(df) >= 5000:
            print(f"  found tabular spike file: {p}")
            return df

    mats = list(session_dir.rglob("*.mat"))
    mat_candidates = []
    for p in mats:
        name = p.name.lower()
        score = 0
        if "spike" in name: score += 3
        if "spk" in name: score += 2
        if "cell" in name: score += 1
        if "clu" in name: score += 1
        mat_candidates.append((score, p))

    for _, p in sorted(mat_candidates, reverse=True):
        df = try_mat(p)
        if df is not None and len(df) >= 5000:
            print(f"  found MATLAB spike file: {p}")
            return df

    raise RuntimeError(f"Could not find a spike event file with >=5000 events under {session_dir}")

for session in SESSIONS:
    print(f"Processing {session}")
    session_dir = RAW_ROOT / session
    outdir = OUT_ROOT / session
    outdir.mkdir(parents=True, exist_ok=True)

    df = find_events(session_dir)
    df = df.sort_values(["time", "actor", "mark"]).reset_index(drop=True)

    # If times appear to be sample indices rather than seconds, do not guess a sampling rate.
    # The paper experiments only require consistent relative event times; downstream 5k extraction
    # rebases the window.
    top_units = (
        df.groupby("actor").size()
          .sort_values(ascending=False)
          .head(20)
          .index.tolist()
    )
    actor_map = {old: new for new, old in enumerate(top_units)}

    df = df[df["actor"].isin(top_units)].copy()
    df["actor"] = df["actor"].map(actor_map).astype(int)
    df["mark"] = 0
    df = df.sort_values(["time", "actor", "mark"]).reset_index(drop=True)
    df["time"] = df["time"] - float(df["time"].min())

    df.to_csv(outdir / "events.csv", index=False)
    pd.DataFrame({"original_actor": list(actor_map.keys()), "actor": list(actor_map.values())}).to_csv(outdir / "unit_map.csv", index=False)

    print(f"  wrote {outdir / 'events.csv'} rows={len(df)} K={df['actor'].nunique()} time=[{df['time'].min():.3f},{df['time'].max():.3f}]")
'@ | python -
```

## 5. Create CRCNS 5k analytic event files and manifests

```powershell
python make_5k_events.py `
  --src data/processed_crcns/Achilles_10252013/events.csv `
  --outdir data/processed_crcns/Achilles_10252013/real_5k `
  --dataset crcns_hc11 `
  --session Achilles_10252013 `
  --condition real_5k `
  --n 5000

python make_5k_events.py `
  --src data/processed_crcns/Cicero_09012014/events.csv `
  --outdir data/processed_crcns/Cicero_09012014/real_5k `
  --dataset crcns_hc11 `
  --session Cicero_09012014 `
  --condition real_5k `
  --n 5000
```

Expected outputs:

```text
data/processed_crcns/Achilles_10252013/real_5k/events.csv
data/processed_crcns/Achilles_10252013/real_5k/manifest.csv
data/processed_crcns/Cicero_09012014/real_5k/events.csv
data/processed_crcns/Cicero_09012014/real_5k/manifest.csv
```

## 6. Create CA1 additive spike-ins and manifest

This command generates the CA1 real-background spike-in files used for Table 5. It preserves real background events and adds planted coordination events with known truth labels.

```powershell
@'
from pathlib import Path
import json
import numpy as np
import pandas as pd

src = Path("data/processed_crcns/Achilles_10252013/real_5k/events.csv")
outdir = Path("data/processed_crcns/Achilles_10252013/spikein_5k_alpha_s20")
outdir.mkdir(parents=True, exist_ok=True)

bg0 = pd.read_csv(src).sort_values("time").reset_index(drop=True)
bg0["is_injected"] = 0
bg0["Z"] = 0

true_ring = [0, 1, 2, 3]
true_hub = 0
active_intervals = [(5, 11), (25, 31), (50, 56), (80, 86)]
alphas = [1.0, 1.5, 2.0, 3.0, 4.0]

manifest_rows = []

for alpha in alphas:
    for seed in range(7, 27):
        rng = np.random.default_rng(seed + int(alpha * 1000))
        injected = []

        # Produces approximately the exposure range reported in the paper
        # while preserving deterministic reproducibility from seed and alpha.
        root_rate = 6.0
        child_prob = 0.80
        delay_mean = 0.25
        delay_max = 1.5

        for s, e in active_intervals:
            n_roots = rng.poisson(root_rate * alpha * (e - s))
            root_times = rng.uniform(s, e, size=n_roots)

            for t in root_times:
                injected.append({
                    "time": float(t),
                    "actor": true_hub,
                    "mark": 0,
                    "is_injected": 1,
                    "Z": 1,
                })

                for actor in [1, 2, 3]:
                    if rng.random() <= child_prob:
                        tc = t + min(rng.exponential(delay_mean), delay_max)
                        if tc < e:
                            injected.append({
                                "time": float(tc),
                                "actor": int(actor),
                                "mark": 0,
                                "is_injected": 1,
                                "Z": 1,
                            })

        add = pd.DataFrame(injected)
        merged = pd.concat([bg0, add], ignore_index=True).sort_values("time").reset_index(drop=True)

        run_dir = outdir / f"alpha_{alpha:g}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        events_path = run_dir / "events.csv"
        truth_path = run_dir / "truth.json"
        intervals_path = run_dir / "active_intervals.csv"

        merged.to_csv(events_path, index=False)
        pd.DataFrame([{"start": s, "end": e} for s, e in active_intervals]).to_csv(intervals_path, index=False)

        truth = {
            "true_ring": true_ring,
            "true_hub": true_hub,
            "alpha1_strength": alpha,
            "seed": seed,
            "n_background_events": int(len(bg0)),
            "n_injected_events": int(len(add)),
            "time_rebased": True,
        }
        truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")

        manifest_rows.append({
            "dataset": "crcns_hc11",
            "session": f"Achilles_10252013_spikein_alpha_{alpha:g}_seed_{seed}",
            "condition": "ca1_spikein",
            "alpha1_strength": alpha,
            "seed": seed,
            "path": str(events_path),
            "truth_json": str(truth_path),
            "active_intervals_csv": str(intervals_path),
            "true_ring": json.dumps(true_ring),
            "true_hub": true_hub,
            "n_background_events": int(len(bg0)),
            "n_injected_events": int(len(add)),
            "events": int(len(merged)),
        })

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(outdir / "manifest.csv", index=False)

print(manifest[["alpha1_strength", "seed", "n_injected_events", "events"]].head().to_string(index=False))
print("WROTE", outdir / "manifest.csv")
'@ | python -
```

## 7. Process Binance raw aggregate trades into q99 minute event stream

This command scans raw Binance aggregate-trade CSV/ZIP files, filters to the 20 symbols and Nov. 6--11, 2022, keeps the largest 1% notional trades within each symbol, converts side to marks, maps symbols to actors, and writes the processed event stream.

The mark convention is:

```text
mark=0: buyer-initiated taker flow, is_buyer_maker=false
mark=1: seller-initiated taker flow, is_buyer_maker=true
```

```powershell
@'
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

RAW = Path("data/raw_crypto/binance_vision")
OUT = Path("data/processed_crypto/binance_vision/ftx_20221106_20221111_q99_minutes")
OUT.mkdir(parents=True, exist_ok=True)

symbols = [
    "FTTUSDT", "SOLUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT",
    "MATICUSDT", "AVAXUSDT", "DOGEUSDT", "ADAUSDT", "XRPUSDT",
    "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "ETCUSDT",
    "ATOMUSDT", "FILUSDT", "AAVEUSDT", "UNIUSDT", "NEARUSDT",
]
sym_to_actor = {s: i for i, s in enumerate(symbols)}

def read_one(path):
    names = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                inner = [n for n in zf.namelist() if n.endswith(".csv")][0]
                with zf.open(inner) as fh:
                    df = pd.read_csv(fh, header=None)
        else:
            df = pd.read_csv(path, header=None)
    except Exception:
        return None

    if df.shape[1] < 7:
        return None

    df = df.iloc[:, :7].copy()
    df.columns = names
    return df

rows = []
for sym in symbols:
    files = []
    for ext in ["*.csv", "*.zip"]:
        files.extend(RAW.rglob(f"*{sym}*{ext.split('*')[-1]}"))
    files = sorted(set(files))

    if not files:
        print(f"WARNING: no raw files found for {sym}")
        continue

    parts = []
    for p in files:
        df = read_one(p)
        if df is None or df.empty:
            continue
        df["symbol"] = sym
        df["date_file"] = p.name
        parts.append(df)

    if not parts:
        print(f"WARNING: no readable rows for {sym}")
        continue

    s = pd.concat(parts, ignore_index=True)
    s["price"] = pd.to_numeric(s["price"], errors="coerce")
    s["quantity"] = pd.to_numeric(s["quantity"], errors="coerce")
    s["transact_time"] = pd.to_numeric(s["transact_time"], errors="coerce")
    s = s.dropna(subset=["price", "quantity", "transact_time"])

    # Keep Nov 6 00:00 UTC through Nov 11 23:59:59 UTC.
    start_ms = int(pd.Timestamp("2022-11-06T00:00:00Z").timestamp() * 1000)
    end_ms = int(pd.Timestamp("2022-11-12T00:00:00Z").timestamp() * 1000)
    s = s[(s["transact_time"] >= start_ms) & (s["transact_time"] < end_ms)].copy()
    if s.empty:
        print(f"WARNING: no rows in target date range for {sym}")
        continue

    s["notional"] = s["price"] * s["quantity"]
    q = s["notional"].quantile(0.99)
    s = s[s["notional"] >= q].copy()

    s["time"] = (s["transact_time"] - start_ms) / 60000.0
    s["actor"] = sym_to_actor[sym]
    bm = s["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"])
    s["mark"] = bm.astype(int)

    rows.append(s[[
        "time", "transact_time", "actor", "mark", "symbol",
        "price", "quantity", "notional", "is_buyer_maker", "date_file",
    ]])

events = pd.concat(rows, ignore_index=True).sort_values(["time", "actor", "mark"]).reset_index(drop=True)
events.to_csv(OUT / "events.csv", index=False)

asset_map = pd.DataFrame({"symbol": symbols, "actor": [sym_to_actor[s] for s in symbols]})
asset_map.to_csv(OUT / "asset_map.csv", index=False)

print("WROTE", OUT / "events.csv", "rows=", len(events))
print("WROTE", OUT / "asset_map.csv")
print(events.groupby("symbol").size().to_string())
'@ | python -
```

## 8. Create FTX two-hour temporal-sweep windows

This command slices the six-day processed Binance q99 stream into the eight rebased two-hour windows used in the FTX temporal sweep.

```powershell
@'
from pathlib import Path
import pandas as pd

src = Path("data/processed_crypto/binance_vision/ftx_20221106_20221111_q99_minutes/events.csv")
outroot = Path("data/processed_crypto/binance_vision/ftx_window_sweep_rebased")
outroot.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src).sort_values("time").reset_index(drop=True)

windows = [
    ("nov06_1200_quiet_anchor",          720,  120),
    ("nov07_0000_early",                1440, 120),
    ("nov07_1200_pre_collapse",         2160, 120),
    ("nov08_0000_market_stress",        2880, 120),
    ("nov08_1600_peak",                 3840, 120),
    ("nov09_0000_post_peak",            4320, 120),
    ("nov10_1600_second_ftt_spike",     6720, 120),
    ("nov11_1400_bankruptcy_sol_heavy", 8040, 120),
]

rows = []
for name, start, duration in windows:
    wdir = outroot / name
    wdir.mkdir(parents=True, exist_ok=True)

    w = df[(df["time"] >= start) & (df["time"] < start + duration)].copy()
    w["time"] = w["time"] - start
    w = w.sort_values("time").reset_index(drop=True)

    events_path = wdir / "events.csv"
    w.to_csv(events_path, index=False)

    rows.append({
        "dataset": "binance_vision",
        "session": name,
        "condition": "ftx_window_sweep_rebased",
        "path": str(events_path),
        "events": len(w),
    })

manifest = pd.DataFrame(rows)
manifest.to_csv(outroot / "manifest.csv", index=False)

print(manifest.to_string(index=False))
print("WROTE", outroot / "manifest.csv")
'@ | python -
```

## 9. Generate FTX receiver-target spike-ins

```powershell
python crypto_additive_spikein_compact.py `
  --events-csv data/processed_crypto/binance_vision/ftx_20221106_20221111_q99_minutes/events.csv `
  --asset-map data/processed_crypto/binance_vision/ftx_20221106_20221111_q99_minutes/asset_map.csv `
  --outdir data/processed_crypto/binance_vision/ftx_spikein_compact_cap5k_s5 `
  --time-start 3840 `
  --duration 120 `
  --background-cap 5000 `
  --alpha-list 1.0,1.5,2.0,3.0,4.0 `
  --seed-start 7 `
  --n-reps 5 `
  --root-rate 6.0 `
  --child-prob 0.90 `
  --delay-mean 3.0 `
  --delay-max 12.0
```

Expected output:

```text
data/processed_crypto/binance_vision/ftx_spikein_compact_cap5k_s5/crypto_spikein_compact_manifest.csv
```

## 10. Synthetic baseline recovery

Scaled K=20 benchmark:

```powershell
python reviewer_upgrade_experiments.py `
  --outdir runs/synthetic_scaled_k20_iter30_s20 `
  --n-reps 20 `
  --seed-start 7 `
  --max-iters 30 `
  --methods proposed,b1,b2,b3
```

Base K=10 benchmark:

```powershell
@'
from pathlib import Path
from regime_hawkes.config import SimConfig
from reviewer_upgrade_experiments import run_replications, parse_method_selection

cfg = SimConfig(
    K=10,
    M=2,
    T=500.0,
    ring_actors=[0, 1, 2],
    hub_actor=0,
    d=3,
    nu_base=0.15,
    alpha0_team=0.01,
    alpha1_max=0.8,
    alpha1_min=0.35,
    beta0=1.0,
    beta1=3.0,
    eta_on=0.06,
    eta_off=0.3,
    seed=7,
)

run_replications(
    cfg,
    seeds=list(range(7, 27)),
    outdir=Path("runs/synthetic_base_k10_iter30_s20"),
    method="topk",
    methods=parse_method_selection("proposed,b1,b2,b3"),
    max_iters=30,
)
'@ | python -
```

## 11. Synthetic design ablations

Scaled K=20 design ablations:

```powershell
python reviewer_upgrade_design_ablations.py `
  --outdir runs/design_ablations_scaled_k20_iter30_s20 `
  --n-reps 20 `
  --seed-start 7 `
  --max-iters 30
```

Base K=10 design ablations:

```powershell
@'
from pathlib import Path
from regime_hawkes.config import SimConfig
from reviewer_upgrade_design_ablations import run_design_ablations

cfg = SimConfig(
    K=10,
    M=2,
    T=500.0,
    ring_actors=[0, 1, 2],
    hub_actor=0,
    d=3,
    nu_base=0.15,
    alpha0_team=0.01,
    alpha1_max=0.8,
    alpha1_min=0.35,
    beta0=1.0,
    beta1=3.0,
    eta_on=0.06,
    eta_off=0.3,
    seed=7,
)

run_design_ablations(
    cfg,
    seeds=list(range(7, 27)),
    outdir=Path("runs/design_ablations_base_k10_iter30_s20"),
    max_iters=30,
)
'@ | python -
```

## 12. Raw CA1 held-out predictive fit

The paper reports the completed paired predictive runs with `n=14` CA1 seeds because these likelihood experiments are the most expensive real-background experiments.

Achilles:

```powershell
python reviewer_upgrade_crcns_hc_advisor_mp_SAFE.py `
  --manifest data/processed_crcns/Achilles_10252013/real_5k/manifest.csv `
  --outdir runs/real_achilles_5k_nll_iter30_s14 `
  --methods proposed,b1,b2,b3 `
  --workers 14 `
  --n-reps 14 `
  --seed-start 7 `
  --max-iters 30 `
  --heldout-frac 0.2 `
  --dt 2.0
```

Cicero:

```powershell
python reviewer_upgrade_crcns_hc_advisor_mp_SAFE.py `
  --manifest data/processed_crcns/Cicero_09012014/real_5k/manifest.csv `
  --outdir runs/real_cicero_5k_nll_iter30_s14 `
  --methods proposed,b1,b2,b3 `
  --workers 14 `
  --n-reps 14 `
  --seed-start 7 `
  --max-iters 30 `
  --heldout-frac 0.2 `
  --dt 2.0
```

## 13. Raw CA1 permutation/equivariance diagnostic

```powershell
python reviewer_upgrade_crcns_hc_advisor_mp_SAFE.py `
  --manifest data/processed_crcns/Achilles_10252013/real_5k/manifest.csv `
  --outdir runs/real_achilles_5k_perm_seed777_proposed_s20 `
  --methods proposed `
  --workers 10 `
  --n-reps 20 `
  --seed-start 777 `
  --max-iters 30 `
  --dt 2.0
```

## 14. CA1 real-background spike-in recovery

```powershell
python reviewer_upgrade_crcns_hc_advisor_mp_SAFE.py `
  --manifest data/processed_crcns/Achilles_10252013/spikein_5k_alpha_s20/manifest.csv `
  --outdir runs/ca1_achilles_spikein_5k_all_methods_iter30_s20 `
  --methods proposed,b1,b2,b3 `
  --workers 14 `
  --use-manifest-seeds `
  --max-iters 30 `
  --dt 2.0
```

## 15. FTX raw temporal sweep

All comparable methods:

```powershell
python reviewer_upgrade_crcns_hc_advisor_mp_SAFE.py `
  --manifest data/processed_crypto/binance_vision/ftx_window_sweep_rebased/manifest.csv `
  --outdir runs/ftx_window_sweep_rebased_all_methods_iter10_s7 `
  --methods proposed,b1,b2,b3 `
  --workers 7 `
  --n-reps 7 `
  --seed-start 7 `
  --max-iters 10 `
  --dt 2.0
```

Proposed-only temporal dynamics:

```powershell
python reviewer_upgrade_crcns_hc_advisor_mp_SAFE.py `
  --manifest data/processed_crypto/binance_vision/ftx_window_sweep_rebased/manifest.csv `
  --outdir runs/ftx_window_sweep_rebased_proposed_iter10_s7 `
  --methods proposed `
  --workers 7 `
  --n-reps 7 `
  --seed-start 7 `
  --max-iters 10 `
  --dt 2.0
```

## 16. FTX receiver-target spike-in recovery

```powershell
python reviewer_upgrade_crcns_hc_advisor_mp_SAFE.py `
  --manifest data/processed_crypto/binance_vision/ftx_spikein_compact_cap5k_s5/crypto_spikein_compact_manifest.csv `
  --outdir runs/ftx_spikein_compact_cap5k_s5_proposed_iter7 `
  --methods proposed `
  --workers 5 `
  --use-manifest-seeds `
  --max-iters 7 `
  --dt 2.0
```

## 17. FTX mark-channel rho recovery

For mark-channel lifts, run with full nuisance updates enabled. This is slower; use one worker.

```powershell
$env:CRCNS_FAST_MSTEP_NUISANCE = "0"

python reviewer_upgrade_crcns_hc_advisor_mp_SAFE.py `
  --manifest data/processed_crypto/binance_vision/ftx_spikein_compact_cap5k_s5/crypto_spikein_compact_manifest.csv `
  --outdir runs/ftx_spikein_compact_cap5k_s5_proposed_iter4_rhoFull `
  --methods proposed `
  --workers 1 `
  --use-manifest-seeds `
  --max-iters 4 `
  --dt 2.0 `
  --max-sessions 30

Remove-Item Env:\CRCNS_FAST_MSTEP_NUISANCE
```

## 18. Paired NLL significance tests

```powershell
@'
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

CA1 = Path("runs/real_achilles_5k_nll_iter30_s14/crcns_baseline_raw.csv")
FTX = Path("runs/ftx_window_sweep_rebased_all_methods_iter10_s7/crcns_baseline_raw.csv")
out_path = Path("paper_outputs/paired_nll_tests.csv")

def paired_tests(path, label, session_filter=None):
    df = pd.read_csv(path)

    if session_filter is not None:
        df = df[df["session"].astype(str).str.contains(session_filter, regex=False)].copy()

    df = df[df["method"].isin(["Proposed", "B1_static_gates", "B3_static_svd"])].copy()

    rows = []
    for baseline in ["B1_static_gates", "B3_static_svd"]:
        wide = df.pivot_table(
            index="seed",
            columns="method",
            values="heldout_nll_per_event",
            aggfunc="first",
        ).dropna(subset=["Proposed", baseline])

        diff = wide[baseline].astype(float) - wide["Proposed"].astype(float)
        n = len(diff)
        mean = diff.mean()
        sd = diff.std(ddof=1)
        se = sd / np.sqrt(n)

        t_stat, t_p = stats.ttest_rel(
            wide[baseline].astype(float),
            wide["Proposed"].astype(float),
        )

        try:
            w = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
            w_stat, w_p = float(w.statistic), float(w.pvalue)
        except Exception:
            w_stat, w_p = np.nan, np.nan

        rows.append({
            "experiment": label,
            "baseline": baseline,
            "n": n,
            "mean_diff_baseline_minus_proposed": mean,
            "sd_diff": sd,
            "se_diff": se,
            "mean_over_se": mean / se if se > 0 else np.nan,
            "paired_t_p_two_sided": float(t_p),
            "wilcoxon_p_one_sided": w_p,
        })

    return pd.DataFrame(rows)

out = pd.concat([
    paired_tests(CA1, "raw Achilles CA1"),
    paired_tests(FTX, "FTX Nov8 16:00 peak", session_filter="nov08_1600_peak"),
], ignore_index=True)

out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)
print(out.to_string(index=False))
print("WROTE", out_path)
'@ | python -
```

## 19. Extract compact paper-output tables

```powershell
@'
from pathlib import Path
import pandas as pd
import numpy as np

outdir = Path("paper_outputs")
outdir.mkdir(parents=True, exist_ok=True)

def mean_sd(x):
    x = pd.to_numeric(x, errors="coerce")
    return x.mean(), x.std(ddof=1)

# Table 4: real-background held-out NLL
real_paths = [
    ("Achilles", Path("runs/real_achilles_5k_nll_iter30_s14/crcns_baseline_raw.csv")),
    ("Cicero", Path("runs/real_cicero_5k_nll_iter30_s14/crcns_baseline_raw.csv")),
    ("FTX_peak", Path("runs/ftx_window_sweep_rebased_all_methods_iter10_s7/crcns_baseline_raw.csv")),
]

rows = []
for exp_name, path in real_paths:
    df = pd.read_csv(path)
    if exp_name == "FTX_peak":
        df = df[df["session"].astype(str).str.contains("nov08_1600_peak", regex=False)].copy()
    for method, g in df.groupby("method"):
        if "heldout_nll_per_event" not in g.columns:
            continue
        nll_m, nll_s = mean_sd(g["heldout_nll_per_event"])
        act_m, act_s = mean_sd(g["active_mean"]) if "active_mean" in g.columns else (np.nan, np.nan)
        rows.append({
            "experiment": exp_name,
            "method": method,
            "n": len(g),
            "nll_mean": nll_m,
            "nll_sd": nll_s,
            "active_mean": act_m,
            "active_sd": act_s,
        })

pd.DataFrame(rows).to_csv(outdir / "table_real_nll.csv", index=False)

# Table 5 top: CA1 spike-in recovery
ca1 = pd.read_csv("runs/ca1_achilles_spikein_5k_all_methods_iter30_s20/crcns_baseline_raw.csv")
ca1_rows = (
    ca1.groupby(["method", "alpha1_strength"], dropna=False)
       .agg(f1_mean=("f1", "mean"), f1_sd=("f1", "std"), n=("f1", "count"))
       .reset_index()
)
ca1_rows.to_csv(outdir / "table_ca1_spikein_recovery.csv", index=False)

# Table 5 bottom: FTX receiver-target recovery
ftx = pd.read_csv("runs/ftx_spikein_compact_cap5k_s5_proposed_iter7/crcns_baseline_raw.csv")
metric_cols = [c for c in ftx.columns if c in ["target_receiver_F1", "source_top2_member", "source_top4_member", "strict_hub", "f1"]]
if metric_cols:
    ftx_rows = (
        ftx.groupby("alpha1_strength", dropna=False)[metric_cols]
           .mean()
           .reset_index()
    )
    ftx_rows.to_csv(outdir / "table_ftx_receiver_spikein.csv", index=False)

# Table 7: FTX temporal sweep
w = pd.read_csv("runs/ftx_window_sweep_rebased_proposed_iter10_s7/crcns_baseline_raw.csv")
cols = ["session", "events", "heldout_nll_per_event", "active_mean",
        "sender0_score", "receiver0_score", "receiver4_score", "receiver1_score"]
present = [c for c in cols if c in w.columns]
temporal = w.groupby("session", dropna=False)[present[1:]].mean().reset_index()
temporal.to_csv(outdir / "table_ftx_temporal_sweep.csv", index=False)

# Table 9: mark-channel lifts
rho = pd.read_csv("runs/ftx_spikein_compact_cap5k_s5_proposed_iter4_rhoFull/crcns_baseline_raw.csv")
rho_cols = [c for c in ["rho_lift_00", "rho_lift_01", "rho_lift_10", "rho_lift_11"] if c in rho.columns]
rho_table = rho.groupby("alpha1_strength", dropna=False)[rho_cols].mean().reset_index()
rho_table.to_csv(outdir / "table_ftx_mark_channels.csv", index=False)

print("WROTE:")
for p in sorted(outdir.glob("table_*.csv")):
    print(" ", p)
'@ | python -
```

## 20. Optional FTX lag-correlation diagnostic

```powershell
@'
from pathlib import Path
import pandas as pd
import numpy as np

src = Path("data/processed_crypto/binance_vision/ftx_20221106_20221111_q99_minutes/events.csv")
out = Path("paper_outputs/ftx_lagcorr_diagnostics.csv")
out.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src)
peak = df[(df["time"] >= 3840) & (df["time"] < 3960)].copy()
peak["bin5"] = ((peak["time"] - 3840) // 5).astype(int)

symbols = sorted(peak["symbol"].dropna().unique())
bins = range(24)

counts = {}
for s in symbols:
    c = peak[peak["symbol"] == s].groupby("bin5").size().reindex(bins, fill_value=0).astype(float).to_numpy()
    counts[s] = c

rows = []
for a in symbols:
    for b in symbols:
        if a == b:
            continue
        best_lag = None
        best_corr = -np.inf
        zero_corr = np.nan
        for lag in [0, 1, 2, 3, 4, 5, 6]:
            x = counts[a][:-lag] if lag > 0 else counts[a]
            y = counts[b][lag:] if lag > 0 else counts[b]
            if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
                corr = np.nan
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
            if lag == 0:
                zero_corr = corr
            if np.isfinite(corr) and corr > best_corr:
                best_corr = corr
                best_lag = lag * 5
        rows.append({
            "pair": f"{a}->{b}",
            "source": a,
            "target": b,
            "best_lag_min": best_lag,
            "best_corr": best_corr,
            "zero_lag_corr": zero_corr,
            "lift_vs_zero": best_corr - zero_corr if np.isfinite(zero_corr) else np.nan,
        })

res = pd.DataFrame(rows).sort_values(["best_corr", "lift_vs_zero"], ascending=False)
res.to_csv(out, index=False)

print("Top delayed pairs:")
print(res[(res["best_lag_min"] > 0)].head(20).to_string(index=False))
print("WROTE", out)
'@ | python -
```

## 21. Cached outputs included in this repository

This repository includes cached outputs only for the real-background experiments,
because those are the expensive runs and depend on public raw-data preprocessing.
The synthetic experiments are generated on the fly from fixed random seeds and
do not require cached data files.

Expected cached real-output folders:

```text
runs/
  real_achilles_5k_nll/
    crcns_baseline_raw.csv
    crcns_baseline_summary.csv

  real_cicero_5k_nll/
    crcns_baseline_raw.csv
    crcns_baseline_summary.csv

  real_achilles_5k_perm_seed777_proposed/
    crcns_baseline_raw.csv
    crcns_baseline_summary.csv

  ca1_achilles_spikein_5k_all_methods/
    crcns_baseline_raw.csv
    crcns_baseline_summary.csv

  ftx_window_sweep_rebased_all_methods_iter10_s7/
    crcns_baseline_raw.csv
    crcns_baseline_summary.csv

  ftx_window_sweep_rebased_proposed_iter10_s7/
    crcns_baseline_raw.csv
    crcns_baseline_summary.csv

  ftx_spikein_compact_cap5k_s5_proposed_iter7/
    crcns_baseline_raw.csv
    crcns_baseline_summary.csv

  ftx_spikein_compact_cap5k_s5_proposed_iter4_rhoFull/
    crcns_baseline_raw.csv
    crcns_baseline_summary.csv

  ucdp_ged_india_volume_neutral_p1_s20/
    ucdp_ged_spikein_raw_checkpoint.csv
    ucdp_ged_spikein_theorem_summary.csv
    ucdp_p1_exposure_table.tex
    ucdp_theorem_paired_deltas.csv

  ucdp_ged_india_weaklink_p2_diag_s20/
    ucdp_ged_spikein_raw_checkpoint.csv
    ucdp_p2_threshold_points.csv
    ucdp_p2_slope_fit.csv
```

The following outputs are intentionally not cached by default and can be
regenerated directly from the commands in this README:

```text
runs/synthetic_base_k10_iter30_s20/
runs/synthetic_scaled_k20_iter30_s20/
runs/design_ablations_base_k10_iter30_s20/
runs/design_ablations_scaled_k20_iter30_s20/
runs/b4_base_synthetic/
runs/ucdp_ged_india_coarsening_diag_s20/
```

These synthetic and design-ablation results are simulated from fixed seeds by
the reproduction scripts and do not require external data files.



## 22. UCDP GED social-science data prep and theorem-facing tests

The UCDP GED experiments use a real conflict-event background and inject a
volume-neutral planted coordination signal. The paper uses the exact-day GED
subset, streams by dyad, and the India 2007--2011 window with `K=8` streams and
`k=4` planted actors.

### 22.1 Prepare the UCDP GED event table

```powershell
python ucdp_ged_events.py `
  --ged-csv data/raw_ucdp/GEDEvent_v26_1.csv `
  --stream-by dyad `
  --violence-types 1 `
  --exact-dates-only `
  --out-prefix data/processed_ucdp/ucdp_ged_dyad_exact_v26_1
```

Expected outputs:

```text
data/processed_ucdp/ucdp_ged_dyad_exact_v26_1_events.csv
data/processed_ucdp/ucdp_ged_dyad_exact_v26_1_streams.npz
data/processed_ucdp/ucdp_ged_dyad_exact_v26_1_streams_index.json
```

### 22.2 Screen candidate windows

This command is a convenience screen for dense dyad windows. The paper run below
also pins the India window explicitly, so the screen is useful for verification
rather than required for the deterministic reproduction path.

```powershell
python screen_ucdp_ged_windows.py `
  --events data/processed_ucdp/ucdp_ged_dyad_exact_v26_1_events.csv `
  --group-by country `
  --country India `
  --window-years 5 `
  --step-years 1 `
  --top-k 8 `
  --min-events 400 `
  --min-topk-events 200 `
  --min-streams 8 `
  --out data/processed_ucdp/ucdp_ged_india_candidate_windows.csv
```

### 22.3 UCDP P1 volume-neutral exposure sweep

This is the real social-science exposure experiment reported in the paper. It
holds the original planted-member background count fixed while realized planted
active-born exposure varies from approximately 25 to 200 events.

```powershell
python -m regime_hawkes.run_ucdp_ged_spikein_mp `
  --events data/processed_ucdp/ucdp_ged_dyad_exact_v26_1_events.csv `
  --country India `
  --window-start 2007-01-01 `
  --window-end 2012-01-01 `
  --top-k 8 `
  --subgroup-size 4 `
  --source-actor 0 `
  --subgroup-mode random `
  --volume-mode neutral `
  --mark-mode fatality_bin `
  --injection-strengths 0.5 1.0 2.0 3.0 4.0 `
  --base-injected-events 50 `
  --coarsen-days 0.0 `
  --seeds 20 `
  --seed-start 0 `
  --workers 8 `
  --methods proposed,b1 `
  --interval-width 7 `
  --jitter 0.05 `
  --n-episodes 4 `
  --episode-length 21 `
  --max-iters 30 `
  --n-inner-steps 10 `
  --outdir runs/ucdp_ged_india_volume_neutral_p1_s20
```

Evaluate the UCDP P1 run and write the paper-ready CSV/LaTeX summaries:

```powershell
python -m regime_hawkes.evaluate_ucdp_ged_spikein `
  --raw runs/ucdp_ged_india_volume_neutral_p1_s20/ucdp_ged_spikein_raw_checkpoint.csv `
  --outdir runs/ucdp_ged_india_volume_neutral_p1_s20 `
  --headline-strength 4.0 `
  --n-boot 2000
```

Expected outputs:

```text
runs/ucdp_ged_india_volume_neutral_p1_s20/ucdp_ged_spikein_raw_checkpoint.csv
runs/ucdp_ged_india_volume_neutral_p1_s20/ucdp_ged_spikein_theorem_summary.csv
runs/ucdp_ged_india_volume_neutral_p1_s20/ucdp_p1_exposure_table.tex
runs/ucdp_ged_india_volume_neutral_p1_s20/ucdp_theorem_paired_deltas.csv
```

### 22.4 UCDP weak-link diagnostic stress test

This real-background diagnostic varies one designated receiver's injected share
while redistributing the freed mass to the other receivers. It is retained as a
stress test; the paper's theorem-facing weakest-link coordinate is the synthetic
true-`alpha_min` sweep.

```powershell
python -m regime_hawkes.run_ucdp_ged_spikein_mp `
  --events data/processed_ucdp/ucdp_ged_dyad_exact_v26_1_events.csv `
  --country India `
  --window-start 2007-01-01 `
  --window-end 2012-01-01 `
  --top-k 8 `
  --subgroup-size 4 `
  --source-actor 0 `
  --subgroup-mode random `
  --volume-mode neutral `
  --mark-mode fatality_bin `
  --injection-strengths 1.0 `
  --base-injected-events 150 `
  --coarsen-days 0.0 `
  --weak-receiver-fracs 1.0 0.75 0.50 0.35 0.25 `
  --episode-lengths 7 14 21 35 56 `
  --seeds 20 `
  --seed-start 0 `
  --workers 8 `
  --methods proposed `
  --interval-width 7 `
  --jitter 0.05 `
  --n-episodes 4 `
  --max-iters 30 `
  --n-inner-steps 10 `
  --outdir runs/ucdp_ged_india_weaklink_p2_diag_s20
```

Evaluate the diagnostic:

```powershell
python -m regime_hawkes.evaluate_ucdp_ged_spikein `
  --raw runs/ucdp_ged_india_weaklink_p2_diag_s20/ucdp_ged_spikein_raw_checkpoint.csv `
  --outdir runs/ucdp_ged_india_weaklink_p2_diag_s20 `
  --n-boot 2000
```

Expected P2 diagnostic outputs:

```text
runs/ucdp_ged_india_weaklink_p2_diag_s20/ucdp_p2_threshold_points.csv
runs/ucdp_ged_india_weaklink_p2_diag_s20/ucdp_p2_slope_fit.csv
```

### 22.5 Optional UCDP temporal-coarsening diagnostic

This diagnostic admits coarser GED date uncertainty levels to stress the
posterior/temporal-localization bridge. It is not used as a theorem-clean P3
claim because the hard-path posterior gap changes with exposure and date
coarsening.

```powershell
python -m regime_hawkes.run_ucdp_ged_spikein_mp `
  --events data/processed_ucdp/ucdp_ged_dyad_exact_v26_1_events.csv `
  --country India `
  --window-start 2007-01-01 `
  --window-end 2012-01-01 `
  --top-k 8 `
  --subgroup-size 4 `
  --source-actor 0 `
  --subgroup-mode random `
  --volume-mode neutral `
  --mark-mode fatality_bin `
  --injection-strengths 1.0 `
  --base-injected-events 150 `
  --coarsen-days 0.0 1.0 3.0 7.0 14.0 `
  --seeds 20 `
  --seed-start 0 `
  --workers 8 `
  --methods proposed `
  --interval-width 7 `
  --jitter 0.05 `
  --n-episodes 4 `
  --episode-length 21 `
  --max-iters 30 `
  --n-inner-steps 10 `
  --outdir runs/ucdp_ged_india_coarsening_diag_s20

python -m regime_hawkes.evaluate_ucdp_ged_spikein `
  --raw runs/ucdp_ged_india_coarsening_diag_s20/ucdp_ged_spikein_raw_checkpoint.csv `
  --outdir runs/ucdp_ged_india_coarsening_diag_s20 `
  --n-boot 2000
```

## 23. External B4 change-point Hawkes baseline on the base synthetic benchmark

The external B4 baseline is a change-point Hawkes pipeline: segment first, then
fit a static Hawkes model inside the selected active segment. The base synthetic
run below uses the same K=10 configuration as Section 10, but selects only B4
and writes to the run folder used in the paper package.

```powershell
@'
from pathlib import Path
from regime_hawkes.config import SimConfig
from reviewer_upgrade_experiments import run_replications, parse_method_selection

cfg = SimConfig(
    K=10,
    M=2,
    T=500.0,
    ring_actors=[0, 1, 2],
    hub_actor=0,
    d=3,
    nu_base=0.15,
    alpha0_team=0.01,
    alpha1_max=0.8,
    alpha1_min=0.35,
    beta0=1.0,
    beta1=3.0,
    eta_on=0.06,
    eta_off=0.3,
    seed=7,
)

run_replications(
    cfg,
    seeds=list(range(7, 27)),
    outdir=Path("runs/b4_base_synthetic"),
    method="topk",
    methods=parse_method_selection("b4"),
    max_iters=30,
)
'@ | python -
```


## 24. Replication-count note

The synthetic and CA1 spike-in recovery experiments use 20 seeds. The UCDP GED volume-neutral exposure and weak-link diagnostic runs use 20 seeds. The raw CA1 held-out predictive likelihood comparison uses the completed paired predictive seed set, `n=14`, due to compute time. The FTX peak predictive comparison uses `n=7`. The financial mark-channel recovery uses 5 spike-in seeds and full nuisance updates for rho estimation.




## 25. Powered sparse directed-edge diagnostic

This port includes the powered sparse directed-edge experiment added after the original repository snapshot. It is documented in `SPARSE_DIRECTED_DIAGNOSTIC.md`. The main commands are:

```powershell
python -m regime_hawkes.run_sparse_directed_powered_mp `
  --benchmark large `
  --large-k 10 `
  --large-K 30 `
  --large-T 900 `
  --seeds 20 `
  --max-iters 30 `
  --workers 8

python -m regime_hawkes.evaluate_sparse_directed_edges_with_baselines `
  --scores .\results\sparse_directed_powered\sparse_directed_scores.csv `
  --truth .\results\sparse_directed_powered\sparse_directed_truth.csv `
  --outdir .\results\sparse_directed_powered
```

The completed diagnostic summary from the reviewer-response run is included in `runs/sparse_directed_powered/sparse_directed_edges_summary.csv`.
