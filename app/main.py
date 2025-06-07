from fastapi import FastAPI
from app.routers import router


def init_app():
    server = FastAPI(title="FastAPI stats server")
    server.include_router(router)

    return server


my_app = init_app()
