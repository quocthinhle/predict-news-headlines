import pandas as pd
from sklearn.utils import shuffle
from utils import remove_special_chars, remove_stopwords, remove_tags, lemmatize_word, convert_lower

def get_train_data():
    dataset = pd.read_csv('../dataset/training-set/crawled-data.csv')
    dataset['Text'] = dataset['title'] + ' ' + dataset['description']
    dataset['Text'] = dataset['Text'].apply(str)
    dataset['CategoryId'] = dataset['Category'].factorize()[0]
    print(dataset.groupby('Category').CategoryId.value_counts())
    dataset['Text'] = dataset['Text'].apply(remove_tags)
    dataset['Text'] = dataset['Text'].apply(remove_special_chars)
    dataset['Text'] = dataset['Text'].apply(convert_lower)
    dataset['Text'] = dataset['Text'].apply(remove_stopwords)
    dataset['Text'] = dataset['Text'].apply(lemmatize_word)
    return dataset
    
def get_predict_data():
    dataset = pd.read_csv('../dataset/predict-data/predict-data.csv')
    dataset['Text'] = dataset['headlines'] + ' ' + dataset['short_description']
    dataset['Text'] = dataset['Text'].apply(str)
    dataset['Text'] = dataset['Text'].apply(remove_tags)
    dataset['Text'] = dataset['Text'].apply(remove_special_chars)
    dataset['Text'] = dataset['Text'].apply(convert_lower)
    dataset['Text'] = dataset['Text'].apply(remove_stopwords)
    dataset['Text'] = dataset['Text'].apply(lemmatize_word)
    return dataset
