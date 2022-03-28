import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
# nltk.download('punkt')
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from utils import remove_special_chars, remove_stopwords, remove_tags, lemmatize_word, convert_lower

def prepare_data():
    dataset = pd.read_csv('../training-set/headlines-train.csv')
    dataset['CategoryId'] = dataset['Category'].factorize()[0]
    dataset.groupby('Category').CategoryId.value_counts().plot(kind="bar", color=["pink", "orange", "red", "yellow", "blue"])
    dataset['Text'] = dataset['Text'].apply(remove_tags)
    dataset['Text'] = dataset['Text'].apply(remove_special_chars)
    dataset['Text'] = dataset['Text'].apply(convert_lower)
    dataset['Text'] = dataset['Text'].apply(remove_stopwords)
    dataset['Text'] = dataset['Text'].apply(lemmatize_word)
    return dataset
    