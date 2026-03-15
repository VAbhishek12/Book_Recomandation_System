import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommendation System", layout="wide")

st.title("📚 Book Recommendation System")

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():

    books = pd.read_csv("Books.csv", encoding="latin-1")
    ratings = pd.read_csv("Ratings.csv", encoding="latin-1")
    users = pd.read_csv("Users.csv", encoding="latin-1")
    pt = pd.read_pickle("pt.pkl")

    return books, ratings, users, pt


books, ratings, users, pt = load_data()

# -------------------------
# Similarity Matrix
# -------------------------
@st.cache_data
def compute_similarity():
    similarity = cosine_similarity(pt)
    return similarity


similarity_score = compute_similarity()


# -------------------------
# Recommendation Function
# -------------------------
def recommend(book_name):

    try:

        index = np.where(pt.index == book_name)[0][0]

        similar_books = sorted(
            list(enumerate(similarity_score[index])),
            key=lambda x: x[1],
            reverse=True
        )[1:6]

        recommended_books = []
        images = []
        authors = []

        for i in similar_books:

            book_title = pt.index[i[0]]

            temp_df = books[books["Book-Title"] == book_title]

            recommended_books.append(book_title)

            images.append(
                temp_df["Image-URL-L"].values[0]
            )

            authors.append(
                temp_df["Book-Author"].values[0]
            )

        return recommended_books, images, authors

    except:
        return [], [], []


# -------------------------
# UI
# -------------------------

st.subheader("🔎 Find Similar Books")

book_list = list(pt.index.values)

selected_book = st.selectbox(
    "Select a Book",
    book_list
)

if st.button("Recommend"):

    with st.spinner("Finding similar books..."):

        books, images, authors = recommend(selected_book)

    if len(books) == 0:

        st.error("No recommendations found.")

    else:

        cols = st.columns(5)

        for i in range(5):

            with cols[i]:

                st.image(images[i])
                st.write(books[i])
                st.caption(authors[i])
