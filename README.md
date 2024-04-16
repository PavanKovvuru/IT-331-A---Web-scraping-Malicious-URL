# Malicious URL Detection using Machine Learning

This project aims to detect malicious URLs using machine learning techniques. It employs the use of Scikit-learn, BeautifulSoup, and Pandas to preprocess and classify URLs into malicious or benign categories.

Features of the code
1.  Web Scraping: The program scrapes links from a given webpage using BeautifulSoup. 
2.  URL Tokenization: The URLs are tokenized to separate feature words from the raw data, helping in better feature extraction.
3. Vectorization: Two types of vectorization techniques are implemented: 
   a. Count Vectorization: Represents the count of each token in the URLs.
   b. TF-IDF Vectorization: Calculates the importance of each token in the URLs based on the Term Frequency-Inverse Document Frequency.
4. Classification: Utilizes Multinomial Naive Bayes for classification due to its effectiveness in text classification tasks.

Requirements
1. Install all these libraries
    Python 3.x
    Pandas
    NumPy
    Matplotlib
    Scikit-learn
    BeautifulSoup
    Requests
Two python code files are created for each method of Vectorization(Countvectorizer.py and TFIDFVectorizer.py) and the Malicious URLs csv should be in same directory and the code files.
