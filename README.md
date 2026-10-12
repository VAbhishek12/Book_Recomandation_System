# Book Recommendation System

Item-based collaborative filtering that recommends books similar to one you already like, built on the Book-Crossing ratings dataset and served as a Streamlit app.

## How it works

1. **Clean** – keep explicit ratings only (rating > 0).
2. **Filter** – keep active readers (≥ 20 ratings) and books rated by at least 15 of them, so each book has enough signal.
3. **Pivot** – build a book × user rating matrix.
4. **Compare** – compute cosine similarity between every pair of books.
5. **Recommend** – return the five books whose rating vectors are closest to the selected title.

The thresholds are constants at the top of [`src/recommender.py`](src/recommender.py).

## Repository layout

```
.
├── app.py                  Streamlit app
├── src/recommender.py      data loading, filtering, similarity, recommend()
├── data/                   Books.csv, Ratings.csv, Users.csv (Book-Crossing)
├── notebooks/              exploratory analysis and the first prototype
└── requirements.txt
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The similarity matrix is built from the CSVs on first launch (a few seconds) and cached for the session.

## Use it as a library

```python
from src.recommender import build

model = build()
model.recommend("A Bend in the Road", n=5)
```

## Dataset

[Book-Crossing](https://www2.informatik.uni-freiburg.de/~cziegler/BX/) – book metadata, user profiles and about 722 k rating rows (282 k explicit, the rest implicit 0s). The app builds its model on roughly 800 well-rated titles.

## Notes

- The notebook in `notebooks/` is the original exploration. Its filtering is looser than the app's, so recommendations will not match exactly.
- Recommendations are only available for titles that pass the filters above.

## Tech

Python · Pandas · NumPy · scikit-learn · Streamlit
