import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
import pandas as pd

# Define the RecommenderNet class with get_config method
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

    def get_config(self):
        return {'num_users': self.user_embedding.input_shape[0],
                'num_items': self.item_embedding.input_shape[0],
                'embedding_size': self.user_embedding.output_shape[-1]}


# Load the model with custom objects and handle unexpected arguments (optional)
model_path = r'C:\Users\ASUS\Desktop\BRS\New folder\book_recommendation_model.h5'
model = load_model(model_path, custom_objects={'RecommenderNet': RecommenderNet},
                  compile=False,
                  # Ignore unexpected arguments during deserialization (optional)
                  load_weights=lambda f, *args, **kwargs: f(*args))

# Loading the data frames
df = pd.read_csv(r"C:\Users\ASUS\Desktop\BRS\New folder\df_for_tf.csv")
db_books = pd.read_csv(r"C:\Users\ASUS\Desktop\BRS\New folder\df_books.csv")


# Function to get book recommendations
def book_recommendation(userid):
    user_ids = df['User-ID'].unique().tolist()
    item_ids = df['ISBN'].unique().tolist()

    user_to_index = {user: idx for idx, user in enumerate(user_ids)}
    item_to_index = {item: idx for idx, item in enumerate(item_ids)}

    df['user'] = df['User-ID'].map(user_to_index)
    df['item'] = df['ISBN'].map(item_to_index)

    user_index = user_to_index[userid]
    item_indices = np.array(list(item_to_index.values()))

    user_item_pairs = np.array([[user_index, item] for item in item_indices])
    predicted_ratings = model.predict([user_item_pairs[:, 0], user_item_pairs[:, 1]])

    recommended_items = {item_ids[item]: predicted_ratings[idx][0] for idx, item in enumerate(item_indices)}
    recommended_items = sorted(recommended_items.items(), key=lambda x: x[1], reverse=True)

    top_5_recommended_items = recommended_items[:5]
    recommended_books = []
    image_urls = []

    for item in top_5_recommended_items:
        recommended_books.append(db_books.loc[db_books['ISBN'] == item[0], 'Book-Title'].values[0])
        image_urls.append(db_books.loc[db_books['ISBN'] == item[0], 'Image-URL-L'].values[0])

    return recommended_books, image_urls


# Streamlit app layout
tab1, tab2 = st.columns(2)

with tab2:
    st.title("Book Recommender System (TensorFlow)")
    user_id_input = st.text_input("Enter User ID", "")

    if st.button('Get Recommendations'):
        if user_id_input:
            try:
                user_int = int(user_id_input)
                book_names, images = book_recommendation(user_int)

                for i in range(len(book_names)):
                    st.image(images[i])
                    st.markdown(book_names[i], unsafe_allow_
