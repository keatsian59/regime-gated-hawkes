import argparse
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--dataset", required=True)
p.add_argument("--session", required=True)
p.add_argument("--condition", default="real_5k")
p.add_argument("--n", type=int, default=5000)
args = p.parse_args()

src = Path(args.src)
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src)
df = df.sort_values(["time", "actor", "mark"]).head(args.n).copy()
t0 = float(df["time"].min())
df["time"] = df["time"] - t0

events_path = outdir / "events.csv"
df.to_csv(events_path, index=False)

manifest = pd.DataFrame([{
    "dataset": args.dataset,
    "session": args.session,
    "condition": args.condition,
    "path": str(events_path)
}])
manifest.to_csv(outdir / "manifest.csv", index=False)

print("Wrote", events_path)
print("Rows", len(df))
print("Time range", float(df["time"].min()), float(df["time"].max()))
print("Wrote", outdir / "manifest.csv")
