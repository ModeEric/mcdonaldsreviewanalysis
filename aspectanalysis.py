import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from textblob import TextBlob
import re
nltk.download('stopwords')
df = pd.read_csv('data/McDonald_s_Reviews.csv', encoding='latin-1')
stop_words = set(stopwords.words('english'))
df['review'] = df['review'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
df['review'] = df['review'].apply(lambda x: re.sub('[^a-zA-Z\s]', '', x))

df['rating'] = df['rating'].apply(lambda x: int(re.search(r'\d+', x).group()))
df['label'] = df['rating'].apply(lambda x: 'negative' if x < 3 else 'positive') 

vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
X_train_vect = vectorizer.fit_transform(df['review'])

negative_reviews = df[df['label'] == 'negative']['review']
positive_reviews = df[df['label'] == 'positive']['review']
vectorizer.fit(negative_reviews)
negative_word_scores = vectorizer.transform(negative_reviews).sum(axis=0)
positive_word_scores = vectorizer.transform(positive_reviews).sum(axis=0)
word_scores = negative_word_scores - positive_word_scores
word_scores_df = pd.DataFrame(word_scores.T, columns=['score'], index=vectorizer.get_feature_names_out())
top_negative_words = word_scores_df.sort_values(by='score', ascending=False).head(10)
print(top_negative_words)
top_positive_words = word_scores_df.sort_values(by='score', ascending=True).head(10)
print(top_positive_words)