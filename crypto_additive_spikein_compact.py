from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

def stratified_cap(df, cap, rng):
    if cap is None or cap <= 0 or len(df) <= cap:
        return df.copy()

    parts = []
    # Preserve symbol/mark composition as much as possible.
    grouped = list(df.groupby(["symbol", "mark"], dropna=False))
    sizes = np.array([len(g) for _, g in grouped], dtype=float)
    alloc = np.floor(cap * sizes / sizes.sum()).astype(int)
    alloc = np.maximum(alloc, 1)

    # Fix rounding overflow.
    while alloc.sum() > cap:
        j = int(np.argmax(alloc))
        if alloc[j] > 1:
            alloc[j] -= 1
        else:
            break

    # Add leftover to largest groups.
    while alloc.sum() < cap:
        j = int(np.argmax(sizes - alloc))
        alloc[j] += 1

    for (key, g), n in zip(grouped, alloc):
        n = min(int(n), len(g))
        parts.append(g.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))

    return pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-csv", required=True)
    ap.add_argument("--asset-map", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--time-start", type=float, default=960.0)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--background-cap", type=int, default=15000)
    ap.add_argument("--alpha-list", default="1.0,1.5,2.0,3.0,4.0")
    ap.add_argument("--seed-start", type=int, default=7)
    ap.add_argument("--n-reps", type=int, default=3)
    ap.add_argument("--root-rate", type=float, default=6.0)
    ap.add_argument("--child-prob", type=float, default=0.90)
    ap.add_argument("--delay-mean", type=float, default=3.0)
    ap.add_argument("--delay-max", type=float, default=12.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ev0 = pd.read_csv(args.events_csv)
    amap = pd.read_csv(args.asset_map)
    amap.to_csv(outdir / "asset_map.csv", index=False)

    sym_to_actor = dict(zip(amap["symbol"], amap["actor"].astype(int)))

    # Financial spike-in truth, aligned with the raw lead-lag diagnostics.
    ring_symbols = ["FTTUSDT", "BNBUSDT", "AVAXUSDT", "UNIUSDT"]
    true_ring = [int(sym_to_actor[s]) for s in ring_symbols]
    true_hub = int(sym_to_actor["FTTUSDT"])

    # Keep only target background window and rebase to [0, duration].
    bg = ev0[(ev0["time"] >= args.time_start) & (ev0["time"] < args.time_start + args.duration)].copy()
    bg["time"] = bg["time"] - args.time_start
    bg["is_injected"] = 0
    bg["Z"] = 0

    manifest_rows = []
    alphas = [float(x) for x in args.alpha_list.split(",") if x.strip()]

    active_intervals = [(5, 11), (25, 31), (50, 56), (80, 86)]

    for alpha in alphas:
        for seed in range(args.seed_start, args.seed_start + args.n_reps):
            rng = np.random.default_rng(seed + int(alpha * 1000))

            bg_cap = stratified_cap(bg, args.background_cap, rng)
            injected = []

            for s, e in active_intervals:
                n_roots = rng.poisson(args.root_rate * alpha * (e - s))
                root_times = rng.uniform(s, e, size=n_roots)

                for t in root_times:
                    injected.append({
                        "time": float(t),
                        "transact_time": np.nan,
                        "actor": true_hub,
                        "mark": 1,
                        "symbol": "FTTUSDT",
                        "price": np.nan,
                        "quantity": np.nan,
                        "notional": np.nan,
                        "is_buyer_maker": "true",
                        "date_file": "synthetic_spikein",
                        "is_injected": 1,
                        "Z": 1,
                    })

                    for sym in ["BNBUSDT", "AVAXUSDT", "UNIUSDT"]:
                        if rng.random() <= args.child_prob:
                            d = min(rng.exponential(args.delay_mean), args.delay_max)
                            tc = t + d
                            if tc < e:
                                injected.append({
                                    "time": float(tc),
                                    "transact_time": np.nan,
                                    "actor": int(sym_to_actor[sym]),
                                    "mark": 1,
                                    "symbol": sym,
                                    "price": np.nan,
                                    "quantity": np.nan,
                                    "notional": np.nan,
                                    "is_buyer_maker": "true",
                                    "date_file": "synthetic_spikein",
                                    "is_injected": 1,
                                    "Z": 1,
                                })

                    if rng.random() <= 0.50 * args.child_prob:
                        tc = t + min(rng.exponential(args.delay_mean * 1.5), args.delay_max)
                        if tc < e:
                            injected.append({
                                "time": float(tc),
                                "transact_time": np.nan,
                                "actor": int(sym_to_actor["UNIUSDT"]),
                                "mark": 1,
                                "symbol": "UNIUSDT",
                                "price": np.nan,
                                "quantity": np.nan,
                                "notional": np.nan,
                                "is_buyer_maker": "true",
                                "date_file": "synthetic_spikein",
                                "is_injected": 1,
                                "Z": 1,
                            })

            add = pd.DataFrame(injected)
            merged = pd.concat([bg_cap, add], ignore_index=True).sort_values("time").reset_index(drop=True)

            run_dir = outdir / f"alpha_{alpha:g}_seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            events_path = run_dir / "events.csv"
            truth_path = run_dir / "truth.json"
            intervals_path = run_dir / "active_intervals.csv"

            merged.to_csv(events_path, index=False)
            pd.DataFrame([{"start": s, "end": e} for s, e in active_intervals]).to_csv(intervals_path, index=False)

            truth = {
                "true_ring": true_ring,
                "true_ring_symbols": ring_symbols,
                "true_hub": true_hub,
                "true_hub_symbol": "FTTUSDT",
                "alpha1_strength": alpha,
                "seed": seed,
                "n_background_events": int(len(bg_cap)),
                "n_injected_events": int(len(add)),
                "background_cap": int(args.background_cap),
                "time_rebased": True,
            }
            truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")

            manifest_rows.append({
                "dataset": "binance_vision",
                "session": f"ftx_k20_q99_minutes_spikein_compact_alpha_{alpha:g}_seed_{seed}",
                "condition": "crypto_spikein_compact",
                "alpha1_strength": alpha,
                "seed": seed,
                "path": str(events_path),
                "truth_json": str(truth_path),
                "active_intervals_csv": str(intervals_path),
                "true_ring": json.dumps(true_ring),
                "true_hub": true_hub,
                "n_background_events": int(len(bg_cap)),
                "n_injected_events": int(len(add)),
                "events": int(len(merged)),
            })

    man = pd.DataFrame(manifest_rows)
    man.to_csv(outdir / "crypto_spikein_compact_manifest.csv", index=False)

    print("WROTE:", outdir / "crypto_spikein_compact_manifest.csv")
    print(man[["alpha1_strength","seed","n_background_events","n_injected_events","events","true_ring","true_hub"]].to_string(index=False))

if __name__ == "__main__":
    main()
