from fastapi import FastAPI

app = FastAPI()


@app.get('/')
def read_root():
    return {'message': 'ML Ops API'}


@app.get('/another-endpoint')
def another_endpoint():
    return {'result': 'something'}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

