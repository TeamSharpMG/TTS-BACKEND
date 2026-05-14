from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers import stt

app = FastAPI()


origins = [
    "http://localhost:5173",  # your frontend (React, Vue, etc.)
    "http://127.0.0.1:5173",
    # your website
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "world"}


app.include_router(mms.router, prefix='/model/mms', tags=["model/mms"])
app.include_router(stt.router)  