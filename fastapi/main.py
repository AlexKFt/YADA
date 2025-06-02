from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse

from pydantic import BaseModel
import storage_management.utils as fm_utils

import search_core.knowledge_base as kb
from storage_management.files_management import TextSector, Reader
from text_processing.query_processing import TextPreparator

app = FastAPI()

app.mount("/static", StaticFiles(directory=r"C:\Users\Alexander\source\yada\Storage"))

FILES_ROOT = r"C:\Users\Alexander\source\yada\Storage"
index_list = fm_utils.get_dir_files(FILES_ROOT, lambda x: x.split('.')[-1] == "faiss")
dict_list = fm_utils.get_dir_files(r"C:\Users\Alexander\source\yada\Storage\extension_dicts", lambda x: x.split('.')[-1] == "json")

knowledge_base = kb.KnowledgeBase()
reader = Reader()
preparator = TextPreparator()


@app.get("/")
def root():
    return RedirectResponse('/docs')


@app.post("/query", response_model=None)
def create_query(data: dict = Body()):
    request = data["query"]

    return {"result": "something about " + request}


@app.post("/sign-up", response_model=None)
def create_user(user: dict = Body()):
    user_name = user["name"]
    user_role = user["password"]
    return {"user_name": user_name, "user_password": user_role}


@app.post("/sign-in", response_model=None)
def get_user(user: dict = Body()):
    user_name = user["name"]
    user_role = user["password"]
    return {"user_name": user_name, "user_password": user_role, "token": "token"}


@app.get("/index_list")
def get_index_list():
    global index_list
    return {"index_list": {i: file[0] for i, file in enumerate(index_list)}}


@app.post("/index", response_model=None)
def set_work_index(data: dict = Body()):
    idx = int(data["id"])
    cur_index = index_list[idx]
    knowledge_base.read_index_file(cur_index[1])
    reader.read_index(cur_index, knowledge_base, FILES_ROOT)

    return cur_index[0]


@app.get("/dictionary")
def get_dictinaries_list():
    global dict_list
    return {"dict_list": {i: file[0] for i, file in enumerate(dict_list)}}


@app.get("/dictionary")
def get_dictionary(id: int):
    global dict_list
    return FileResponse(dict_list[id][1], filename=index_list[id][0])


@app.post("/dictionary", response_model=None)
def add_work_dictinary(data: dict = Body()):
    id = data["id"]
    return dict_list[id][0]


@app.delete("/dictionary")
def delete_work_dictinary(data: dict =  Body()):
    id = data["id"]
    return dict_list[id][0]


@app.get("/file/{idx}", response_class=FileResponse)
def get_file(id: int):
    return FileResponse(index_list[id][1], filename=index_list[id][0])


class User(BaseModel):
    name : str
    role : str
    id : int