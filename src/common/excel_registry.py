import pandas as pd
from pathlib import Path
from typing import List, Dict

REQUIRED_COLS = [
    "id","person_id","split","src_path","aligned_path","bbox",
    "landmarks","embedding_path","embedding_dim",
    "detector","encoder","hash","ts"
]

def read_or_create_xlsx(xlsx_path: Path) -> pd.DataFrame:
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
        # make sure columns exist
        for c in REQUIRED_COLS:
            if c not in df.columns: df[c] = ""
        return df
    else:
        df = pd.DataFrame(columns=REQUIRED_COLS)
        df.to_excel(xlsx_path, index=False)
        return df

def append_rows(xlsx_path: Path, rows: List[Dict]) -> None:
    df = read_or_create_xlsx(xlsx_path)
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_excel(xlsx_path, index=False)

def update_df(xlsx_path: Path, df: pd.DataFrame) -> None:
    # keep col order
    for c in REQUIRED_COLS:
        if c not in df.columns: df[c] = ""
    df.to_excel(xlsx_path, index=False)
