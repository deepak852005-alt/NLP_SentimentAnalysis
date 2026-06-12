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

# ------------------------------
# Average Rating Over Time
# ------------------------------

st.header("📆 Average Rating Over Time")

df['date'] = pd.to_datetime(
    df['date'],
    errors='coerce'
)

df = df.dropna(subset=['date'])

avg_rating = (
    df.groupby('date')['rating']
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    avg_rating['date'],
    avg_rating['rating']
)

ax.set_title("Average Rating Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Average Rating")

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