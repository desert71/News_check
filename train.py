import pickle
import requests
import bz2
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

resp = requests.get('https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2')
with open('lenta-ru-news.csv.bz2', 'wb') as f:
    f.write(resp.content)



with bz2.BZ2File('lenta-ru-news.csv.bz2') as f:
    content = f.read()
with open('lenta-ru-news.csv', 'wb') as file:
    file.write(content)


df = pd.read_csv("lenta-ru-news.csv", delimiter=',', low_memory=False)
df_test = df.copy(deep=True)
ds_test = df_test.dropna()
x = np.array(ds_test["text"])
y = np.array(ds_test["topic"])


cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
with open('final_model_and_vectorizer.pkl', 'wb') as fout:
    pickle.dump((cv, model), fout)
#joblib.dump(model, "./model.joblib")
