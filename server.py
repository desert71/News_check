import uvicorn
import pickle
import requests
from fastapi import FastAPI
from pydantic import HttpUrl
from bs4 import BeautifulSoup as b

app = FastAPI()
#model = joblib.load("./model.joblib")
with open('final_model_and_vectorizer.pkl', 'rb') as f:
    cv, model = pickle.load(f)

@app.get("/")
def index():
    return "Для классификации по заголовку добавьте в тело запроса /titles/ЗАГОЛОВОК; для классификации по адресу страницы добавьте в тело запроса /urls/АДРЕС_СТРАНИЦЫ"


@app.get("/titles/{title}")
def classify_text(title: str):
    dataa = cv.transform([title]).toarray()
    output = model.predict(dataa)
    scor = model.predict_proba(dataa)
    return f'Категория новостей: {output[0]}, вероятность: {scor.max()}'


@app.get("/urls/")
def classify_url(url:HttpUrl):
    c = pars_title(url)
    title = c[0]
    dataa = cv.transform([title]).toarray()
    output = model.predict(dataa)
    scor = model.predict_proba(dataa)
    return f'Категория новостей: {output[0]}, вероятность: {scor.max()}'

def pars_title(url):
    r = requests.get(url)
    soup = b(r.text, 'html.parser')
    title = soup.find_all('h1', class_='article__heading article__heading_article-page')
    return [c.text for c in title]



if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=8000)