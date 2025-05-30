from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    html = "<h2>My first python web app!</h2>"
    return HTMLResponse(content=html)

@app.get("/request/{id}")
def create_request(id : int):
    #business login
    return {"response" : f"There should be some complicated dictionary"}

@app.post("/sign_in")
def get_user():
    return {"user_name" : "?", "user_role" : "?"}

class User(BaseModel):
    name : str
    role : str
    id : int