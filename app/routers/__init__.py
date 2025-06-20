from fastapi import APIRouter
from .urls import urls_router


router = APIRouter()
router.include_router(urls_router, prefix="/urls")
