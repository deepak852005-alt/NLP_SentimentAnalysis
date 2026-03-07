import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load Model
model = pickle.load(open("sentiment_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# Load Dataset
df = pd.read_csv(r"C:\Users\deepak\OneDrive\Desktop\NLP_SentimentAnalysis\chatgpt_style_reviews_dataset.csv")

# Sentiment column recreate
def sentiment_label(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

df["sentiment"] = df["rating"].apply(sentiment_label)

st.title("🤖 AI Echo - ChatGPT Review Sentiment Dashboard")

# ------------------------------
# Sentiment Prediction
# ------------------------------

st.header("🔍 Predict Review Sentiment")

review = st.text_area("Enter a review")

if st.button("Predict Sentiment"):

    vec = vectorizer.transform([review])
    prediction = model.predict(vec)

    st.success(f"Predicted Sentiment: {prediction[0]}")

# ------------------------------
# Rating Distribution
# ------------------------------

st.header("📊 Rating Distribution")

fig, ax = plt.subplots()

df["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)

plt.xlabel("Rating")
plt.ylabel("Count")

st.pyplot(fig)

# ------------------------------
# Sentiment Distribution
# ------------------------------

st.header("😊 Sentiment Distribution")

fig, ax = plt.subplots()

df["sentiment"].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax)

st.pyplot(fig)

# ------------------------------
# Platform vs Sentiment
# ------------------------------

st.header("💻 Platform vs Sentiment")

fig, ax = plt.subplots()

sns.countplot(x="platform", hue="sentiment", data=df, ax=ax)

st.pyplot(fig)

# ------------------------------
# Verified Users vs Sentiment
# ------------------------------

st.header("✅ Verified Users vs Sentiment")

fig, ax = plt.subplots()

sns.countplot(x="verified_purchase", hue="sentiment", data=df, ax=ax)

st.pyplot(fig)

# ------------------------------
# WordCloud Positive
# ------------------------------

st.header("🌟 Positive Review WordCloud")

positive = df[df["sentiment"]=="Positive"]

text = " ".join(positive["review"])

wordcloud = WordCloud(width=800,height=400,background_color="white").generate(text)

fig, ax = plt.subplots()

ax.imshow(wordcloud)

ax.axis("off")

st.pyplot(fig)

# ------------------------------
# WordCloud Negative
# ------------------------------

st.header("⚠ Negative Review WordCloud")

negative = df[df["sentiment"]=="Negative"]

text = " ".join(negative["review"])

wordcloud = WordCloud(width=800,height=400,background_color="white").generate(text)

fig, ax = plt.subplots()

ax.imshow(wordcloud)

ax.axis("off")

st.pyplot(fig)