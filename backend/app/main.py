import os
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

# from starlette.middleware.sessions import SessionMiddleware
from routers import document_upload_router
import uvicorn

app = FastAPI()


# noinspection PyTypeChecker
def create_app() -> FastAPI:
    # container = Container()
    # container.database().create_database()

    # default_projects = container.config.default.projects()
    # all_project_codes = [
    #     project.project_code for project in container.project_service().get_all()
    # ]
    # for project_code in default_projects:
    #     if project_code not in all_project_codes:
    #         container.project_service().create(project_code)

    # default_email = container.config.default.admin.email()
    # if container.account_service().get_by_email(default_email) is None:
    #     default_password = container.config.default.admin.password()
    #     container.account_service().create(
    #         default_email,
    #         default_password,
    #         ACCOUNT_ADMIN,
    #         ACCOUNT_ENABLED,
    #         default_projects[0],
    #     )

    # container.vectorization_scheduler().start_all_unfinished_doc_files()

    fast_api = FastAPI()
    # fast_api.container = container
    fast_api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add SessionMiddleware
    # fast_api.add_middleware(SessionMiddleware, secret_key="session-secret")

    fast_api.include_router(document_upload_router.router)

    return fast_api


app = create_app()
