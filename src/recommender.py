"""Item-based collaborative filtering for books.

Pipeline
--------
1. Keep explicit ratings only (rating > 0).
2. Keep active users (>= MIN_USER_RATINGS ratings) and well-rated-by-many
   books (>= MIN_BOOK_RATINGS ratings) so the user-item matrix is dense enough
   for cosine similarity to be meaningful.
3. Pivot to a (book x user) matrix and compute cosine similarity between books.
4. Recommend the books whose rating vectors are closest to the selected one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MIN_USER_RATINGS = 20
MIN_BOOK_RATINGS = 15


@dataclass
class Recommender:
    books: pd.DataFrame          # one row per title: title, author, image
    pivot: pd.DataFrame          # book x user rating matrix
    similarity: np.ndarray       # book x book cosine similarity

    @property
    def titles(self) -> list[str]:
        return list(self.pivot.index)

    def recommend(self, title: str, n: int = 5) -> pd.DataFrame:
        """Return the `n` most similar books to `title` (empty if unknown)."""
        if title not in self.pivot.index:
            return self.books.iloc[0:0]
        idx = self.pivot.index.get_loc(title)
        order = np.argsort(self.similarity[idx])[::-1]
        order = [i for i in order if i != idx][:n]
        out = self.books.loc[[self.pivot.index[i] for i in order]].copy()
        out["similarity"] = [self.similarity[idx][i] for i in order]
        return out.reset_index()


def load_raw(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    books = pd.read_csv(data_dir / "Books.csv", encoding="latin-1", low_memory=False)
    ratings = pd.read_csv(data_dir / "Ratings.csv", encoding="latin-1")
    return books, ratings


def build(data_dir: Path = DATA_DIR) -> Recommender:
    books, ratings = load_raw(data_dir)

    ratings = ratings[ratings["Book-Rating"] > 0]

    active_users = ratings["User-ID"].value_counts()
    active_users = active_users[active_users >= MIN_USER_RATINGS].index
    ratings = ratings[ratings["User-ID"].isin(active_users)]

    merged = ratings.merge(books.drop_duplicates("ISBN"), on="ISBN")

    popular = merged["Book-Title"].value_counts()
    popular = popular[popular >= MIN_BOOK_RATINGS].index
    merged = merged[merged["Book-Title"].isin(popular)]

    pivot = merged.pivot_table(
        index="Book-Title", columns="User-ID", values="Book-Rating", aggfunc="mean"
    ).fillna(0)

    catalogue = (
        merged.drop_duplicates("Book-Title")
        .set_index("Book-Title")[["Book-Author", "Image-URL-L"]]
        .rename(columns={"Book-Author": "author", "Image-URL-L": "image"})
        .loc[pivot.index]
    )
    catalogue.index.name = "title"

    return Recommender(catalogue, pivot, cosine_similarity(pivot.values))
