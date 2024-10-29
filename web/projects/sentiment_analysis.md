# Advanced Sentiment Analysis with Movie Reviews

This Python script performs an advanced sentiment analysis on movie reviews using the NLTK (Natural Language Toolkit) library and the Scikit-learn machine learning library.

## Introduction
This script aims to build an advanced sentiment analysis model using the movie reviews dataset from the NLTK corpus. The model is based on the Naive Bayes classification algorithm and is trained on a combination of text features, including bag-of-words (Count Vectorizer) and term frequency-inverse document frequency (TF-IDF Vectorizer). Additionally, the script incorporates sentiment analysis using the VADER (Valence Aware Dictionary and sEntiment Reasoner) library to provide a sentiment score for each review.

## Setup
1. Ensure that the necessary libraries are installed:
   ```python
   # !pip install nltk pandas scikit-learn
   ```
2. Download the required NLTK resources:
   ```python
   nltk.download("movie_reviews")
   nltk.download("vader_lexicon")
   ```

## Data Preparation
1. Load the movie reviews dataset from the NLTK corpus:
   ```python
   documents = [(" ".join(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]
   ```
2. Convert the data to a DataFrame:
   ```python
   df = pd.DataFrame(documents, columns=["review", "sentiment"])
   ```

## Feature Engineering
### Count Vectorizer
1. Create a Count Vectorizer with a maximum of 5,000 features and remove English stop words:
   ```python
   count_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
   X_count = count_vectorizer.fit_transform(df["review"])
   ```

### TF-IDF Vectorizer
1. Create a TF-IDF Vectorizer with a maximum of 5,000 features and remove English stop words:
   ```python
   tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
   X_tfidf = tfidf_vectorizer.fit_transform(df["review"])
   ```

### Sentiment Analysis
1. Use the VADER (Valence Aware Dictionary and sEntiment Reasoner) library to calculate the sentiment score for each review:
   ```python
   sia = SentimentIntensityAnalyzer()
   df["sentiment_score"] = df["review"].apply(lambda x: sia.polarity_scores(x)["compound"])
   ```

## Model Training
### Splitting the Data
1. Split the data into training and testing sets:
   ```python
   X_train_count, X_test_count, y_train, y_test = train_test_split(X_count, df["sentiment"], test_size=0.2, random_state=42)
   X_train_tfidf, X_test_tfidf, _, _ = train_test_split(X_tfidf, df["sentiment"], test_size=0.2, random_state=42)
   ```

### Tuning the Naive Bayes Model
1. Perform a grid search to find the best hyperparameters for the Multinomial Naive Bayes model:
   ```python
   param_grid = {"alpha": [0.1, 1, 10]}
   grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring="f1_macro")
   grid_search.fit(X_train_count, y_train)
   best_model = grid_search.best_estimator_
   ```

### Evaluating the Model
1. Evaluate the model's performance using the test data:
   ```python
   y_pred_count = best_model.predict(X_test_count)
   y_pred_tfidf = best_model.predict(X_test_tfidf)

   print("Count Vectorizer:")
   print(f"Accuracy: {accuracy_score(y_test, y_pred_count)}")
   print(f"F1 Score: {f1_score(y_test, y_pred_count, average='macro')}")
   print(f"Precision: {precision_score(y_test, y_pred_count, average='macro')}")
   print(f"Recall: {recall_score(y_test, y_pred_count, average='macro')}")
   print(f"Classification Report:\n{classification_report(y_test, y_pred_count)}")

   print("\nTF-IDF Vectorizer:")
   print(f"Accuracy: {accuracy_score(y_test, y_pred_tfidf)}")
   print(f"F1 Score: {f1_score(y_test, y_pred_tfidf, average='macro')}")
   print(f"Precision: {precision_score(y_test, y_pred_tfidf, average='macro')}")
   print(f"Recall: {recall_score(y_test, y_pred_tfidf, average='macro')}")
   print(f"Classification Report:\n{classification_report(y_test, y_pred_tfidf)}")
   ```

## Sentiment Prediction
1. Define a function to predict the sentiment of a given text:
   ```python
   def predict_sentiment(text):
       text_vector = tfidf_vectorizer.transform([text])
       prediction = best_model.predict(text_vector)
       sentiment_score = sia.polarity_scores(text)["compound"]
       return prediction[0], sentiment_score
   ```

## Saving the Model and Vectorizers
1. Save the trained model and vectorizers to disk for future use:
   ```python
   with open("model.pkl", "wb") as f:
       pickle.dump(best_model, f)
   with open("count_vectorizer.pkl", "wb") as f:
       pickle.dump(count_vectorizer, f)
   with open("tfidf_vectorizer.pkl", "wb") as f:
       pickle.dump(tfidf_vectorizer, f)
   ```

## Testing the Prediction Function
1. Test the `predict_sentiment` function with some sample text:
   ```python
   print(predict_sentiment("I absolutely loved this movie! It was fantastic."))
   print(predict_sentiment("It was a terrible film. I hated it."))
   print(predict_sentiment("The movie was okay, nothing special."))
   ```

The script provides a comprehensive sentiment analysis solution that combines various text feature extraction techniques, sentiment analysis, and model tuning to achieve robust performance on the movie reviews dataset. The saved model and vectorizers can be easily reused for future sentiment analysis tasks.
