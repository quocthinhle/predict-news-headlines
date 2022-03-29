import pandas as pd
from sklearn.utils import shuffle
from utils import remove_special_chars, remove_stopwords, remove_tags, lemmatize_word, convert_lower

def normalize_data():
    dataset = pd.read_csv('../training-set/headlines-train.csv')
    dataset['CategoryId'] = dataset['Category'].factorize()[0]
    dataset['Text'] = dataset['Text'].apply(str)
    dataset.groupby('Category').CategoryId.value_counts().plot(kind="bar", color=["pink", "orange", "red", "yellow", "blue"])
    dataset['Text'] = dataset['Text'].apply(remove_tags)
    dataset['Text'] = dataset['Text'].apply(remove_special_chars)
    dataset['Text'] = dataset['Text'].apply(convert_lower)
    dataset['Text'] = dataset['Text'].apply(remove_stopwords)
    dataset['Text'] = dataset['Text'].apply(lemmatize_word)
    return dataset

def get_test_data():
    dataset = pd.read_csv('../samples/NewsCategorizer.csv')
    # dataset['Text'] = dataset['title'] + ' ' + dataset['description']
    dataset['Text'] = dataset['Text'].apply(str)
    dataset['CategoryId'] = dataset['Category'].factorize()[0]
    print(dataset.groupby('Category').CategoryId.value_counts())
    dataset['Text'] = dataset['Text'].apply(remove_tags)
    dataset['Text'] = dataset['Text'].apply(remove_special_chars)
    dataset['Text'] = dataset['Text'].apply(convert_lower)
    dataset['Text'] = dataset['Text'].apply(remove_stopwords)
    dataset['Text'] = dataset['Text'].apply(lemmatize_word)
    dataset = shuffle(dataset)
    return dataset
    
def get_test_data_1():
    dataset = pd.read_csv('../samples/1533data.csv')
    # dataset['Text'] = dataset['title'] + ' ' + dataset['description']
    dataset['Text'] = dataset['Text'].apply(str)
    dataset['Text'] = dataset['Text'].apply(remove_tags)
    dataset['Text'] = dataset['Text'].apply(remove_special_chars)
    dataset['Text'] = dataset['Text'].apply(convert_lower)
    dataset['Text'] = dataset['Text'].apply(remove_stopwords)
    dataset['Text'] = dataset['Text'].apply(lemmatize_word)
    dataset = shuffle(dataset)
    return dataset