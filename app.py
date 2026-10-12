import streamlit as st

from src.recommender import MIN_BOOK_RATINGS, MIN_USER_RATINGS, build

st.set_page_config(page_title="Book Recommendation System", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Building the similarity matrix (first run only)...")
def get_model():
    return build()


model = get_model()

st.title("📚 Book Recommendation System")
st.caption(
    "Item-based collaborative filtering with cosine similarity on the Book-Crossing "
    f"ratings. {len(model.titles):,} books rated by many active readers "
    f"(books with ≥ {MIN_BOOK_RATINGS} ratings from users with ≥ {MIN_USER_RATINGS})."
)

selected = st.selectbox("Pick a book you liked", model.titles)

if st.button("Recommend", type="primary"):
    recs = model.recommend(selected, n=5)
    if recs.empty:
        st.error("No recommendations found for this title.")
    else:
        for col, (_, row) in zip(st.columns(len(recs)), recs.iterrows()):
            with col:
                st.image(row["image"], use_container_width=True)
                st.markdown(f"**{row['title']}**")
                st.caption(f"{row['author']} · similarity {row['similarity']:.2f}")
