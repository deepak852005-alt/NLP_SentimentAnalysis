# 🤖 AI Echo – ChatGPT Review Sentiment Analysis Dashboard

## 📌 Project Overview

AI Echo is a sentiment analysis system that analyzes user reviews of a ChatGPT application.
Using Natural Language Processing (NLP) and Machine Learning techniques, the system classifies user reviews into **Positive, Neutral, or Negative sentiments**.

The project also includes an interactive **Streamlit dashboard** to visualize review insights and predict sentiment in real time.

---

## 🎯 Objectives

* Analyze customer feedback from ChatGPT reviews
* Classify reviews into Positive, Neutral, or Negative sentiments
* Visualize sentiment trends and user feedback insights
* Provide an interactive dashboard for real-time sentiment prediction

---

## 📊 Dataset Information

The dataset contains **500 user reviews** of a ChatGPT application with the following attributes:

| Column            | Description                       |
| ----------------- | --------------------------------- |
| date              | Date when the review was posted   |
| title             | Short summary of the review       |
| review            | Full user review text             |
| rating            | User rating (1–5 stars)           |
| username          | Reviewer username                 |
| helpful_votes     | Number of helpful votes           |
| review_length     | Length of the review text         |
| platform          | Platform used (Web / Mobile)      |
| language          | Language of the review            |
| location          | Country of the user               |
| version           | ChatGPT version                   |
| verified_purchase | Indicates if the user is verified |

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn
* WordCloud
* Streamlit

### Machine Learning Model

* Logistic Regression

### NLP Techniques

* Text Cleaning
* Stopword Removal
* Tokenization
* TF-IDF Feature Extraction

---

## 🔄 Project Workflow

Dataset Collection
↓
Data Cleaning & Preprocessing
↓
Exploratory Data Analysis (EDA)
↓
Text Preprocessing using NLP
↓
Feature Extraction using TF-IDF
↓
Machine Learning Model Training
↓
Model Evaluation
↓
Streamlit Dashboard Deployment

---

## 📈 Features of the Dashboard

* 🔍 Real-time Sentiment Prediction
* 📊 Rating Distribution Visualization
* 😊 Sentiment Distribution Analysis
* 💻 Platform vs Sentiment Comparison
* ✅ Verified vs Non-Verified Review Analysis
* ☁ Positive and Negative Word Clouds

---

## 📊 Model Performance

Evaluation Metrics Used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The Logistic Regression model achieved **high accuracy in classifying user review sentiments.**

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/AI-Echo-Sentiment-Analysis.git
```

### 2️⃣ Navigate to the Project Folder

```bash
cd AI-Echo-Sentiment-Analysis
```

### 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

The dashboard will open in your browser.

---

## 📷 Dashboard Preview

Interactive Streamlit dashboard showing sentiment prediction and data visualizations.

---

## 📌 Future Improvements

* Use Deep Learning models such as LSTM or BERT
* Add real-time API integration for live review data
* Deploy the application on AWS or Streamlit Cloud
* Implement topic modeling for deeper feedback analysis

---

## 👨‍💻 Author

**Deepak**
B.Sc Computer Science Student

---

## ⭐ Acknowledgment

This project was developed as part of a **Data Science / AI learning project** to demonstrate practical implementation of NLP and Machine Learning techniques for customer sentiment analysis.
