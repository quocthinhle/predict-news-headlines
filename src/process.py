# Create bag words
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from utils import tokenize_and_stem
from classification import prepare_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

def run_model(model_name, est_c, est_pnlty):
    perform_list = []
    mdl = KNeighborsClassifier(n_neighbors=10 , metric= 'minkowski' , p = 4)
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

def tf(x):
    for i in range(len(x)):
        current_max = len(x[i])
        for k in range(len(x[i])):
            x[i][k] = (x[i][k] * 1.0) / float(current_max)


if __name__ == "__main__":
    dataset = prepare_data()
    # cv = CountVectorizer(max_features = 3000)
    print(dataset)
    print("===============================")
    y = np.array(dataset.CategoryId.values)
    # x = cv.fit_transform(dataset.Text).toarray()
    # tf(x)

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0, shuffle = True)
    # run_model('K Nearest Neighbour', est_c=None, est_pnlty=None)
    tfidf_vectorizer = TfidfVectorizer(max_features=200000,
                                       use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset.Text)
    df_tfidfvect = pd.DataFrame(data = tfidf_matrix.toarray())
    # df_tfidfvect = x
    print("df_tfid")
    print(df_tfidfvect)

    x_train, x_test, y_train, y_test = train_test_split(df_tfidfvect, y, test_size = 0.3, random_state = 0, shuffle = True)
    # run_model('K Nearest Neighbour', est_c=None, est_pnlty=None)
    classifier = KNeighborsClassifier(n_neighbors=10 , metric= 'minkowski' , p = 4)
    y_pred1 = tfidf_vectorizer.fit_transform(['Hour ago, I contemplated retirement for a lot of reasons. I felt like people were not sensitive enough to my injuries. I felt like a lot of people were backed, why not me? I have done no less. I have won a lot of games for the team, and I am not feeling backed, said Ashwin'])
    classifier.fit(x_train, y_train)
    yy = classifier.predict(y_pred1)
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
