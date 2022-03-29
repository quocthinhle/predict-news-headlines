import pandas as pd
from sklearn.utils import shuffle

data = pd.read_csv('../samples/NewsCategorizer.csv')
data = shuffle(data)
data = data.head(10000)
data['CategoryId'] = data['Category'].factorize()[0]
data = data.query('Category in ["BUSINESS", "FOOD & DRINK", "ENTERTAINMENT", "SPORTS", "POLITICS"]')
data.to_csv('../samples/1533data.csv')




