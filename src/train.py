# Create bag words
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from normalize import get_train_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

def run_model(model_name, est_c, est_pnlty):
    perform_list = []
    mdl = KNeighborsClassifier(n_neighbors=10 , metric= 'minkowski', p = 2)
    oneVsRest = OneVsRestClassifier(mdl)
    oneVsRest.fit(x_train, y_train)
    y_pred = oneVsRest.predict(x_test)
    # Performance metrics
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    # Get precision, recall, f1 scores
    precision, recall, f1score, support = score(y_test, y_pred, average='micro')
    print(f'Test Accuracy Score of Basic {model_name}: % {accuracy}')
    print(f'Precision : {precision}')
    perform_list.append(dict([
        ('Model', model_name),
        ('Test Accuracy', round(accuracy, 2)),
    ]))

if __name__ == "__main__":
    dataset = get_train_data()
    y = np.array(dataset.CategoryId.values)
    tfidf_vectorizer = TfidfVectorizer(max_features=20000, use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset.Text)
    df_tfidfvect = pd.DataFrame(data = tfidf_matrix.toarray())
    x_train, x_test, y_train, y_test = train_test_split(df_tfidfvect, y, test_size = 0.15, random_state = 0, shuffle = True)
    print()
    print()
    run_model('K Nearest Neighbour', est_c=None, est_pnlty=None)
    
