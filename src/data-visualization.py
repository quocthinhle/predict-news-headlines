import pandas as pd

df = pd.read_csv('../dataset/output/result.csv')
print(df[3000:3020])