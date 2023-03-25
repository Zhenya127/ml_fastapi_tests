from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline("sentiment-analysis")


@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/api")
async def api():
    return await [
        {
            "path": "/",
            "description": "greetings"
        },
        {
            "path": "/predict/",
            "description": "defenition of tonality",
            "request": "I like machine learning!",
            "response": "POSITIVE"
        }
    ]

@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]
