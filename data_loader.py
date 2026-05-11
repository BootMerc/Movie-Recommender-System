import os
import zipfile
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
#  Constants 
DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
ML_URL     = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_PATH   = DATA_DIR / "ml-latest-small.zip"
DATASET_DIR = DATA_DIR / "ml-latest-small"


#  Helpers 

def _download_with_progress(url: str, dest: Path) -> None:
    """Stream-download a file and show a tqdm progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        desc=f"Downloading {dest.name}",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))


def download_dataset(force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_DIR.exists() and not force:
        print(f"[data_loader] Dataset already present at '{DATASET_DIR}'. Skipping download.")
        return

    print("[data_loader] Downloading MovieLens Latest Small dataset …")
    _download_with_progress(ML_URL, ZIP_PATH)

    print("[data_loader] Extracting …")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    ZIP_PATH.unlink(missing_ok=True)  # clean up the zip
    print(f"[data_loader] Dataset ready at '{DATASET_DIR}'.")


#  Loaders 

def load_ratings() -> pd.DataFrame:
    path = DATASET_DIR / "ratings.csv"
    df = pd.read_csv(path, dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int})
    print(f"[data_loader] Ratings loaded  — {len(df):,} rows, "
          f"{df['userId'].nunique():,} users, {df['movieId'].nunique():,} movies.")
    return df


def load_movies() -> pd.DataFrame:
    path = DATASET_DIR / "movies.csv"
    df = pd.read_csv(path, dtype={"movieId": int, "title": str, "genres": str})
    print(f"[data_loader] Movies loaded   — {len(df):,} rows.")
    return df


def load_tags() -> pd.DataFrame:
    path = DATASET_DIR / "tags.csv"
    df = pd.read_csv(path, dtype={"userId": int, "movieId": int, "tag": str, "timestamp": int})
    print(f"[data_loader] Tags loaded     — {len(df):,} rows.")
    return df


def load_links() -> pd.DataFrame:
    path = DATASET_DIR / "links.csv"
    df = pd.read_csv(path, dtype={"movieId": int})
    print(f"[data_loader] Links loaded    — {len(df):,} rows.")
    return df


def load_all() -> dict:
    download_dataset()
    return {
        "ratings": load_ratings(),
        "movies" : load_movies(),
        "tags"   : load_tags(),
        "links"  : load_links(),
    }


#  Quick sanity check 
if __name__ == "__main__":
    data = load_all()
    print("\n── Sample ratings ──")
    print(data["ratings"].head())
    print("\n── Sample movies ──")
    print(data["movies"].head())
