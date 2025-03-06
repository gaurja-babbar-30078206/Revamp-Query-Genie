import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import document_upload_router
import uvicorn

# ✅ Ensure `root_path="/api"` is passed correctly
def create_app() -> FastAPI:
    fast_api = FastAPI(root_path="/api")  # ✅ Now `root_path="/api"` is applied inside `create_app()`
    
    fast_api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    fast_api.include_router(document_upload_router.router)  # ✅ Include routers under `/api`

    return fast_api  # ✅ Return correctly initialized FastAPI app

# ✅ Now `app` is correctly initialized with `/api`
app = create_app()

 