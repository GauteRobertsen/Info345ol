"""
INFO345 – Step 0: reduce the full RateBeer ratings.csv to a smaller working file.

Run this ONCE from the repo folder (inside the "RateBeer Dataset" folder):
    python 00_prepare_data.py

It reads ratings.csv in chunks (so it works with little RAM) and writes:
  ratings_2012_2017.parquet   – ratings from 2012–2017, numeric columns only
  stats_full/                 – small summary tables over the FULL dataset (for Task 1)
"""
import os
import time
import pandas as pd

# the data files are in the parent folder ("RateBeer Dataset")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "ratings.csv")
OUT = os.path.join(HERE, "ratings_2012_2017.parquet")
STATS = os.path.join(HERE, "stats_full")
os.makedirs(STATS, exist_ok=True)

START = pd.Timestamp("2012-01-01").timestamp()
END = pd.Timestamp("2018-01-01").timestamp()

COLS = ["beer_id", "user_id", "date", "appearance", "aroma",
        "palate", "taste", "overall", "rating"]
DTYPES = {"beer_id": "int64", "user_id": "int64", "date": "int64",
          "rating": "float32"}

kept = []
per_beer, per_user, per_year, hist = [], [], [], []
n_total = 0
t0 = time.time()

for chunk in pd.read_csv(SRC, usecols=COLS, chunksize=1_000_000,
                         dtype=DTYPES, on_bad_lines="skip"):
    for c in ["appearance", "aroma", "palate", "taste", "overall"]:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("float32")
    chunk = chunk.dropna(subset=["rating"])
    n_total += len(chunk)

    # summary statistics over the full dataset
    per_beer.append(chunk.groupby("beer_id").size())
    per_user.append(chunk.groupby("user_id").size())
    year = pd.to_datetime(chunk["date"], unit="s").dt.year
    per_year.append(year.value_counts())
    hist.append(chunk["rating"].round(1).value_counts())

    # keep 2012–2017 for modelling
    kept.append(chunk[(chunk["date"] >= START) & (chunk["date"] < END)])
    print(f"{n_total:,} ratings read ({time.time() - t0:.0f}s)")

def combine(parts, name, idx):
    s = pd.concat(parts).groupby(level=0).sum().sort_index()
    s.rename_axis(idx).rename("n").to_csv(os.path.join(STATS, name))

combine(per_beer, "ratings_per_beer.csv", "beer_id")
combine(per_user, "ratings_per_user.csv", "user_id")
combine(per_year, "ratings_per_year.csv", "year")
combine(hist, "rating_histogram.csv", "rating")

df = pd.concat(kept, ignore_index=True).drop_duplicates(["user_id", "beer_id"], keep="last")
df["beer_id"] = df["beer_id"].astype("int32")
df["user_id"] = df["user_id"].astype("int32")
df.to_parquet(OUT, index=False, compression="zstd")

print(f"\nDone. Full dataset: {n_total:,} ratings")
print(f"Kept 2012–2017: {len(df):,} ratings -> {OUT} "
      f"({os.path.getsize(OUT) / 1e6:.0f} MB)")
