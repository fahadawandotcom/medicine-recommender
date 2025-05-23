import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# Load and prepare data
data = pd.read_csv('med_data.csv')
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['symptom'])
model = NearestNeighbors(n_neighbors=1, metric='cosine')
model.fit(X)

# Prediction function
def recommend_medicine(user_symptom):
    user_vec = vectorizer.transform([user_symptom])
    distance, index = model.kneighbors(user_vec)
    return data.iloc[index[0][0]]['medicine']

# Streamlit UI
st.set_page_config(page_title="AI Medicine Recommender", layout="centered")
st.title("💊 AI-Based Medicine Recommender")
st.write("Enter your symptom and get a medicine recommendation.")
symptom_input = st.text_input("Enter symptom:", "")

if st.button("Recommend"):
    if symptom_input.strip() == "":
        st.warning("Please enter a symptom.")
    else:
        medicine = recommend_medicine(symptom_input)
        st.success(f"Recommended Medicine: **{medicine}**")
