from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.mlops_project.model import Model
import pickle
import os

def load_model():
    
    model = Model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code: load model weights, open DB connections, warm caches, etc.
    print("Startup: Hello (service is starting)")
    yield
    # Shutdown code: close DB connections, flush logs, cleanup temp files, etc.
    print("Shutdown: Goodbye (service is stopping)")

app = FastAPI(
    title="Titanic MLOps Deployment API",
    version="0.1.0",
    description="A combined FastAPI script showing common API patterns for deployment.",
    lifespan=lifespan,
)


@app.get('/')
def read_root():
    return {'message': 'ML Ops API'}




@app.get('/another-endpoint')
def another_endpoint():
    return {'result': 'something'}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}