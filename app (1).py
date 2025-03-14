import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Import the CFRecommender class from the recommender.py file

tab1, tab2 = st.tabs(["Colaberative", "Tensorflow",])

# Loading the data frames
df = pd.read_csv(r"C:\Users\ASUS\Desktop\BRS\New folder\df_for_tf.csv")
db_books = pd.read_csv(r"C:\Users\ASUS\Desktop\BRS\New folder\df_books.csv")
#pt = pd.read_csv(r"C:\Users\Abhinav\pt.csv")
pt = pd.read_pickle(r"C:\Users\ASUS\Desktop\BRS\New folder\pt.pkl")

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
    
# Load the model
model = keras.models.load_model('book_recommendation_model', custom_objects={'RecommenderNet': RecommenderNet})



def book_recommendation(userid):

    user_ids = df['User-ID'].unique().tolist()
    item_ids = df['ISBN'].unique().tolist()

    user_to_index = {user: idx for idx, user in enumerate(user_ids)}
    item_to_index = {item: idx for idx, item in enumerate(item_ids)}

    df['user'] = df['User-ID'].map(user_to_index)
    df['item'] = df['ISBN'].map(item_to_index)

    user_index = user_to_index[userid]  # For a specific user id.
    item_indices = np.array(list(item_to_index.values()))

    user_item_pairs = np.array([[user_index, item] for item in item_indices])
    predicted_ratings = model.predict([user_item_pairs[:, 0], user_item_pairs[:, 1]])

    recommended_items = {item_ids[item]: predicted_ratings[idx][0] for idx, item in enumerate(item_indices)}
    recommended_items = sorted(recommended_items.items(), key=lambda x: x[1], reverse=True)

    top_5_recommended_items = recommended_items[:5]
    recommended_books = []
    image_url = []
    authors = []
    
    # Image-URL-S
    
    # print("Top 5 recommended books for this user:")
    for item in top_5_recommended_items:
     #   print(db_books.loc[db_books['ISBN'] == item[0], 'Book-Title'].values[0]) 
        recommended_books.append(db_books.loc[db_books['ISBN'] == item[0], 'Book-Title'].values[0])
        image_url.append(db_books.loc[db_books['ISBN'] == item[0], 'Image-URL-L'].values[0])
        authors.append(db_books.loc[db_books['ISBN'] == item[0], 'Book-Author'].values[0])
     #   return(print("Recommendations: ", db_books.loc[db_books['ISBN'] == item[0], 'Book-Title'].values[0]))
    return recommended_books, image_url, authors

user_int = None

s_score = cosine_similarity(pt)

def recommend(book_name):
    recomm_books_colaberative = []
    image_url = []
    authors = []
    try:
        ind = np.where(pt.index==book_name)[0][0]
        similar_books = sorted(list(enumerate(s_score[ind])), key=lambda x:x[1], reverse=True)[1:6]
        for i in similar_books:
            recomm_books_colaberative.append(pt.index[i[0]])
            image_url.append(db_books.loc[db_books['Book-Title'] == pt.index[i[0]], 'Image-URL-L'].values[0])
            authors.append(db_books.loc[db_books['Book-Title'] == pt.index[i[0]], 'Book-Author'].values[0])
        return recomm_books_colaberative, image_url, authors
    except IndexError:
        return("No Recommendations found for this book")

with tab2:
# Streamlit app
    st.title("Book Recommender System (TensorFlow)")
    user_id_input = st.text_input("Enter User ID", "")

    result=""

    if st.button('Get Recommendations'):
        if user_id_input:
            try:
                user_int = int(user_id_input)
                book_name, images, author = book_recommendation(user_int)
                col1, col2, col3, col4,col5 = st.columns(5)
                with col1:
                    st.image(images[0])
                    st.markdown(book_name[0], unsafe_allow_html=True)
                    st.markdown(author[0], unsafe_allow_html=True)
                with col2:
                    st.image(images[1])
                    st.markdown(book_name[1], unsafe_allow_html=True)
                    st.markdown(author[1], unsafe_allow_html=True)
                with col3:
                    st.image(images[2])
                    st.markdown(book_name[2], unsafe_allow_html=True)
                    st.markdown(author[2], unsafe_allow_html=True)
                with col4:
                    st.image(images[3])
                    st.markdown(book_name[3], unsafe_allow_html=True)
                    st.markdown(author[3], unsafe_allow_html=True)
                with col5:
                    st.image(images[4])
                    st.markdown(book_name[4], unsafe_allow_html=True)
                    st.markdown(author[4], unsafe_allow_html=True)

            
            except ValueError:
                st.error("Please enter a valid User Id")
    
with tab1:
    st.title("Book Recommender System (Colaberative)")
    user_book_name = st.text_input("Enter a book name", "")

    if st.button('Get Recommendations', key="colab"):
        recomm_books, images, author = recommend(user_book_name)
        col1, col2, col3, col4,col5 = st.columns(5)

        with col1:
            st.image(images[0])
            st.markdown(recomm_books[0], unsafe_allow_html=True)
            st.markdown(author[0], unsafe_allow_html=True)
        with col2:
            st.image(images[1])
            st.markdown(recomm_books[1], unsafe_allow_html=True)
            st.markdown(author[1], unsafe_allow_html=True)
        with col3:
            st.image(images[2])
            st.markdown(recomm_books[2], unsafe_allow_html=True)
            st.markdown(author[2], unsafe_allow_html=True)
        with col4:
            st.image(images[3])
            st.markdown(recomm_books[3], unsafe_allow_html=True)
            st.markdown(author[3], unsafe_allow_html=True)
        with col5:
            st.image(images[4])
            st.markdown(recomm_books[4], unsafe_allow_html=True)
            st.markdown(author[4], unsafe_allow_html=True)


    # result = user_id_input
# st.success(book_name)
