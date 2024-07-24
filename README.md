Personal Information Name: Sai Deepika S Background: Student in an engineering college Interests: Learning new technologies, fast-paced learner Domain : Machine Learning ID : CT6ML501

### Project Overview: Sentiment Analysis of Movie Reviews

**Objective**: Develop a sentiment analysis model to classify movie reviews as positive or negative using the IMDb Movie Reviews dataset.

### Steps Involved

1. **Data Collection**:
   - Use the IMDb Movie Reviews dataset from `tensorflow.keras.datasets`.

2. **Data Preprocessing**:
   - Clean the text data by removing HTML tags, special characters, and stopwords.
   - Convert the text to lowercase.

3. **Data Splitting**:
   - Split the dataset into training, validation, and test sets.

4. **Text Vectorization**:
   - Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text data into numerical features.

5. **Model Selection**:
   - Choose a Logistic Regression model for its simplicity and effectiveness in text classification tasks.

6. **Model Training**:
   - Train the Logistic Regression model on the training data.

7. **Model Evaluation**:
   - Evaluate the model using accuracy, precision, recall, and F1 score on the validation set.

8. **Prediction**:
   - Implement a function to predict the sentiment of new reviews based on the trained model.

### Detailed Implementation

**1. Data Collection**:
- Load the IMDb dataset using `tensorflow.keras.datasets.imdb`.

**2. Data Preprocessing**:
- Decode the integer-encoded reviews back to text.
- Remove HTML tags and special characters.
- Convert text to lowercase.
- Remove stopwords using NLTK.

**3. Data Splitting**:
- Split the cleaned dataset into training and validation sets using `train_test_split` from `sklearn`.

**4. Text Vectorization**:
- Convert the text data into numerical features using TF-IDF vectorization.

**5. Model Selection and Training**:
- Train a Logistic Regression model using the TF-IDF features.

**6. Model Evaluation**:
- Evaluate the model's performance on the validation set using accuracy, precision, recall, and F1 score metrics.

**7. Prediction Function**:
- Implement a function that preprocesses a new review, vectorizes it, and predicts its sentiment.

### Code Implementation

```python
# Install required libraries
!pip install tensorflow scikit-learn nltk

# Import libraries
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import nltk
from nltk.corpus import stopwords

# Load IMDb dataset
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
max_features = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Convert to DataFrame for easier handling
word_index = imdb.get_word_index()
index_word = {v: k for k, v in word_index.items()}

def decode_review(encoded_review):
    return ' '.join([index_word.get(i - 3, '?') for i in encoded_review])

train_df = pd.DataFrame({
    'text': [decode_review(review) for review in X_train],
    'label': y_train
})

test_df = pd.DataFrame({
    'text': [decode_review(review) for review in X_test],
    'label': y_test
})

# Preprocess the text data
def preprocess_text(text):
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(train_df['text'], train_df['label'], test_size=0.2, random_state=42)
X_test = test_df['text']
y_test = test_df['label']

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=max_features)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Model Selection and Training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluation
# Validation set predictions
y_val_pred = model.predict(X_val_tfidf)

# Evaluation metrics
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print(f"Validation Accuracy: {accuracy}")
print(f"Validation Precision: {precision}")
print(f"Validation Recall: {recall}")
print(f"Validation F1 Score: {f1}")

# Prediction
def predict_sentiment(review):
    review = preprocess_text(review)
    review_tfidf = vectorizer.transform([review])
    sentiment = model.predict(review_tfidf)[0]
    return "Positive" if sentiment == 1 else "Negative"

# Test the function with a new review
new_review = "The movie was fantastic! I really enjoyed it."
print(predict_sentiment(new_review))
```

### Conclusion

This project involves loading and preprocessing the IMDb movie reviews dataset, vectorizing the text data using TF-IDF, training a logistic regression model, evaluating its performance, and using it to predict the sentiment of new reviews. This end-to-end process provides a comprehensive overview of building a sentiment analysis model for text classification tasks.

![image](https://github.com/user-attachments/assets/8a7e1bd0-ea22-4a45-ba3b-1cffe4253766)
