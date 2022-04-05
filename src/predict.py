# Create bag words
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from normalize import get_predict_data, get_train_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

def classify(x):
    if x == 0:
        return "business"
    elif x == 1:
        return "travel"
    elif x == 2:
        return "sports"
    elif x == 3:
        return "life"
    elif x == 4:
        return "world"

if __name__ == "__main__":
    dataset = get_train_data()
    y_train = np.array(dataset.CategoryId.values)
    tfidf_vectorizer = TfidfVectorizer(max_features=20000, use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset.Text)
    x_train = pd.DataFrame(data = tfidf_matrix.toarray())

    print('===================')

    predict_data = get_predict_data()
    
    classifier = KNeighborsClassifier(n_neighbors=10 , metric= 'minkowski' , p = 4)
    classifier.fit(x_train, y_train)

    x_predict = tfidf_vectorizer.transform(predict_data.Text.values)
    y_predict = pd.DataFrame(data = x_predict.toarray())
    
    result = classifier.predict(y_predict)

    result = list(map(classify, result))

    predict_data['predict_category'] = result
    predict_data.to_csv('../dataset/output/result.csv')
