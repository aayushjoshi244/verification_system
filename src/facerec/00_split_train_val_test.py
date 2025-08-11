import pandas as pd
from pathlib import Path
from src.common.paths import FACE_XLSX

MIN_PER_PERSON = 4          # skip identities with fewer than this many embedded samples
TRAIN_PER_PERSON = 3        # how many go to train
VAL_PER_PERSON = 1          # optional, can be 0
TEST_PER_PERSON = 2         # how many go to test

def main():
    xlsx = FACE_XLSX
    df = pd.read_excel(xlsx)

    # keep only rows with a real embedding_path
    ep = df["embedding_path"].astype(str).str.strip()
    mask_emb = ep.ne("") & ep.str.lower().ne("nan")
    df = df[mask_emb].copy()

    # reset split
    df["split"] = ""

    grouped = df.groupby("person_id", group_keys=False)
    rows = []
    for pid, g in grouped:
        g = g.sample(frac=1.0, random_state=42)  # shuffle deterministically
        if len(g) < MIN_PER_PERSON:
            # too few samples → put all to train (or leave empty if you prefer)
            g["split"] = "train"
        else:
            t = TRAIN_PER_PERSON
            v = VAL_PER_PERSON
            s = TEST_PER_PERSON
            n = len(g)
            t = min(t, n)
            v = min(v, max(0, n - t))
            s = min(s, max(0, n - t - v))
            # assign
            g.iloc[:t, g.columns.get_loc("split")] = "train"
            g.iloc[t:t+v, g.columns.get_loc("split")] = "val" if v > 0 else "train"
            g.iloc[t+v:t+v+s, g.columns.get_loc("split")] = "test"
            # any leftovers → train
            if t+v+s < n:
                g.iloc[t+v+s:, g.columns.get_loc("split")] = "train"
        rows.append(g)

    out = pd.concat(rows, ignore_index=True) if rows else df
    out.to_excel(xlsx, index=False)
    print("Splits written back to", xlsx)
    print(out["split"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
