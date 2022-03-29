# Create bag words
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from utils import tokenize_and_stem
from normalize import normalize_data, get_test_data, get_test_data_1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

def run_model(model_name, est_c, est_pnlty):
    perform_list = []
    mdl = KNeighborsClassifier()
    oneVsRest = OneVsRestClassifier(mdl)
    oneVsRest.fit(x_train, y_train)
    y_pred = oneVsRest.predict(x_test)
    print(x_test)
    print("==========================")
    print(y_pred)
    # Performance metrics
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    # Get precision, recall, f1 scores
    precision, recall, f1score, support = score(y_test, y_pred, average='micro')
    print(f'Test Accuracy Score of Basic {model_name}: % {accuracy}')
    print(f'Precision : {precision}')
    print(f'Recall : {recall}')
    print(f'F1-score : {f1score}')
    perform_list.append(dict([
        ('Model', model_name),
        ('Test Accuracy', round(accuracy, 2)),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(f1score, 2))
    ]))

if __name__ == "__main__":
    dataset = normalize_data()
    print(dataset.CategoryId)
    y = np.array(dataset.CategoryId.values)
    tfidf_vectorizer = TfidfVectorizer(max_features=20000, use_idf=True)
    # cv = CountVectorizer(max_features=20000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset.Text)
    df_tfidfvect = pd.DataFrame(data = tfidf_matrix.toarray())
    x_train, x_test, y_train, y_test = train_test_split(df_tfidfvect, y, test_size = 0.3, random_state = 0, shuffle = True)
    # run_model('K Nearest Neighbour', est_c=None, est_pnlty=None)
    
    # x_train = df_tfidfvect
    # y_train = y

    # print('===================')

    # data_test = get_test_data()
    # print(data_test)
    # test_tfidf = tfidf_vectorizer.transform(data_test.Text)
    # test_tfidf_vect = pd.DataFrame(data = test_tfidf.toarray())

    # x_test = test_tfidf_vect
    # y_test = np.array(data_test.CategoryId.values)
    
    classifier = KNeighborsClassifier(n_neighbors=10 , metric= 'minkowski' , p = 4)
    classifier.fit(x_train, y_train)

    y_pred1 = tfidf_vectorizer.transform(["Russian Igor Frolov has joined the HCMC New Group and will compete at the annual national cycling championship this year."])
    ypredict = pd.DataFrame(data = y_pred1.toarray())
    
    yy = classifier.predict(ypredict)
    result = ""
    if yy == [0]:
        result = "Business News"
    elif yy == [1]:
        result = "Tech News"
    elif yy == [2]:
        result = "Politics News"
    elif yy == [3]:
        result = "Sports News"
    elif yy == [1]:
        result = "Entertainment News"
    print(result)
