# read data from CSV file
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re
from joblib import dump
from joblib import load

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Import Scikit-learn helper functions
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Import Scikit-learn models
#scikit-learn is a free software machine learning library for the Python programming language
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Import Scikit-learn metric functions
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split

print("Imported all necessary packages")

# Tokenize the URL
#   The purpose of a tokenizer is to separate the features from the raw data
def tokenizer(url):
  """Separates feature words from the raw data
  Keyword arguments:
    url ---- The full URL

  :Returns -- The tokenized words; returned as a list
  """
  # Split by slash (/) and dash (-)
  tokens = re.split('[/-]', url)

  for i in tokens:
    # Include the splits extensions and subdomains
    if i.find(".") >= 0:
      dot_split = i.split('.')

      # Remove .com and www. since they're too common
      if "com" in dot_split:
        dot_split.remove("com")
      if "www" in dot_split:
        dot_split.remove("www")

      tokens += dot_split

  return tokens

#print("Done tokenizer")

def vectorizationCount (train_df, test_df):
    # Vectorizer the training inputs
    #   There are two types of vectors:
    #     1. Count vectorizer
    #     2. Term Frequency-Inverse Document Frequency (TF-IDF)
    #     TF-IDF is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents (corpus).
    #     It considers both the term frequency (how often a word appears in a document) and the inverse document frequency (how unique or rare a word is across the entire corpus).
    print("Training Count Vectorizer")
    cVec = CountVectorizer(tokenizer=tokenizer)
    count_X = cVec.fit_transform(train_df['URLs'])

    # Vectorize the testing inputs

    print("Test Count Vectorizer")
    test_count_X = cVec.transform(test_df['URLs'])#   Use 'transform' instead of 'fit_transform' since we had already trained our vectorizers



    print("Vectorizing Completed")
    return  count_X , test_count_X

def vectorizationTFIDF (train_df, test_df):
    # Vectorizer the training inputs
    #   There are two types of vectors:
    #     1. Count vectorizer
    #     2. Term Frequency-Inverse Document Frequency (TF-IDF)
    #     TF-IDF is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents (corpus).
    #     It considers both the term frequency (how often a word appears in a document) and the inverse document frequency (how unique or rare a word is across the entire corpus).

    print("Training TF-IDF Vectorizer")
    tVec = TfidfVectorizer(tokenizer=tokenizer)
    tfidf_X = tVec.fit_transform(train_df['URLs'])


    print("Test TFIDF Vectorizer")
    test_tfidf_X = tVec.transform(test_df['URLs'])

    print("Vectorizing Completed")
    return  test_tfidf_X


def get_links_from_webpage(url):
    # Send a GET request to the webpage
    response = requests.get(url)

    # Check if the GET request is successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all the anchor tags
        anchor_tags = soup.find_all('a')

        # Extract the href attribute from each anchor tag
        links = [urljoin(url, tag['href']) for tag in anchor_tags]

        return links
    else:
        print(f"Failed to load webpage with status code {response.status_code}")
        return []

def  TestvectorizationCount (train_df,):
    # Vectorizer the training inputs
    #   There are two types of vectors:
    #     1. Count vectorizer
    #     2. Term Frequency-Inverse Document Frequency (TF-IDF)
    #     TF-IDF is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents (corpus).
    #     It considers both the term frequency (how often a word appears in a document) and the inverse document frequency (how unique or rare a word is across the entire corpus).
    print("Training Count Vectorizer")
    cVec = CountVectorizer(tokenizer=tokenizer)
    count_X = cVec.fit_transform(train_df['URLs'])

    print("Vectorizing Completed")
    return  count_X 

url_df = pd.read_csv('Malicious URLs.csv')
#url_df.tail(50)
test_percentage = .2
train_df, test_df = train_test_split(url_df, test_size=test_percentage, random_state=42)

labels = train_df['Class']
test_labels = test_df['Class']
print("Seperated training and test data")

tfidf_X, test_tfidf_X =vectorizationCount(train_df, test_df)
#tfidf_X, test_tfidf_X =vectorizationTFIDF(train_df, test_df)
# Train the model
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(tfidf_X, labels)


# Example usage
url = input("Enter the URL: ")
links = get_links_from_webpage(url)
links=pd.DataFrame(links)
links.columns=['URLs']
#finallinks=links.drop(0, axis='columns')
#finallinks=links.to_string(index=False)
links.to_csv('scrapper.csv',index=False)
train_X, test_tfidf_X =vectorizationCount(train_df, links)
predictions_mnb_tfidf = mnb_tfidf.predict(test_tfidf_X )
predictions_mnb_tfidf=pd.DataFrame(predictions_mnb_tfidf)
predictions_mnb_tfidf.columns=['Outcomes']
predictions_mnb_tfidf.to_csv('Results.csv',index=False)
#print(predictions_mnb_tfidf)htt




