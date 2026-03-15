import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommender System")

# -----------------------------
# Load Data
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("df_for_tf.csv")
    db_books = pd.read_csv("df_books.csv")
    pt = pd.read_pickle("pt.pkl")
    return df, db_books, pt

df, db_books, pt = load_data()


# -----------------------------
# Tensorflow Model
# -----------------------------

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size=50):
        super(RecommenderNet, self).__init__()

        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_size)
        self.dot = tf.keras.layers.Dot(axes=1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[0])
        item_vector = self.item_embedding(inputs[1])
        return self.dot([user_vector, item_vector])


@st.cache_resource
def load_model():
    model = keras.models.load_model(
        "book_recommendation_model",
        custom_objects={"RecommenderNet": RecommenderNet}
    )
    return model

model = load_model()

# -----------------------------
# Tensorflow Recommendation
# -----------------------------

def book_recommendation(userid):

    user_ids = df['User-ID'].unique().tolist()
    item_ids = df['ISBN'].unique().tolist()

    user_to_index = {user: idx for idx, user in enumerate(user_ids)}
    item_to_index = {item: idx for idx, item in enumerate(item_ids)}

    if userid not in user_to_index:
        st.error("User ID not found in dataset")
        return [], [], []

    df['user'] = df['User-ID'].map(user_to_index)
    df['item'] = df['ISBN'].map(item_to_index)

    user_index = user_to_index[userid]
    item_indices = np.array(list(item_to_index.values()))

    user_item_pairs = np.array([[user_index, item] for item in item_indices])

    predicted_ratings = model.predict(
        [user_item_pairs[:, 0], user_item_pairs[:, 1]],
        verbose=0
    )

    recommended_items = {
        item_ids[item]: predicted_ratings[idx][0]
        for idx, item in enumerate(item_indices)
    }

    recommended_items = sorted(
        recommended_items.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_books = recommended_items[:5]

    recommended_books = []
    images = []
    authors = []

    for item in top_books:
        try:
            recommended_books.append(
                db_books.loc[
                    db_books['ISBN'] == item[0],
                    'Book-Title'
                ].values[0]
            )

            images.append(
                db_books.loc[
                    db_books['ISBN'] == item[0],
                    'Image-URL-L'
                ].values[0]
            )

            authors.append(
                db_books.loc[
                    db_books['ISBN'] == item[0],
                    'Book-Author'
                ].values[0]
            )

        except:
            continue

    return recommended_books, images, authors


# -----------------------------
# Collaborative Filtering
# -----------------------------

@st.cache_data
def compute_similarity():
    return cosine_similarity(pt)

s_score = compute_similarity()


def recommend(book_name):

    if book_name not in pt.index:
        st.error("Book not found in dataset")
        return [], [], []

    recomm_books = []
    images = []
    authors = []

    ind = np.where(pt.index == book_name)[0][0]

    similar_books = sorted(
        list(enumerate(s_score[ind])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    for i in similar_books:
        title = pt.index[i[0]]

        recomm_books.append(title)

        images.append(
            db_books.loc[
                db_books['Book-Title'] == title,
                'Image-URL-L'
            ].values[0]
        )

        authors.append(
            db_books.loc[
                db_books['Book-Title'] == title,
                'Book-Author'
            ].values[0]
        )

    return recomm_books, images, authors


# -----------------------------
# Streamlit UI
# -----------------------------

tab1, tab2 = st.tabs(["Collaborative Filtering", "TensorFlow Model"])


# -----------------------------
# TAB 1
# -----------------------------

with tab1:

    st.title("Book Recommender System (Collaborative)")

    book_name = st.text_input("Enter Book Name")

    if st.button("Recommend Books"):

        books, images, authors = recommend(book_name)

        if books:

            cols = st.columns(5)

            for i in range(len(books)):
                with cols[i]:
                    st.image(images[i])
                    st.write(books[i])
                    st.write(authors[i])


# -----------------------------
# TAB 2
# -----------------------------

with tab2:

    st.title("Book Recommender System (TensorFlow)")

    user_id_input = st.text_input("Enter User ID")

    if st.button("Get Personalized Recommendations"):

        try:

            user_id = int(user_id_input)

            books, images, authors = book_recommendation(user_id)

            if books:

                cols = st.columns(5)

                for i in range(len(books)):
                    with cols[i]:
                        st.image(images[i])
                        st.write(books[i])
                        st.write(authors[i])

        except:
            st.error("Please enter a valid numeric User ID")
