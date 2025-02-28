import os
from fastapi import APIRouter
from models.models import Domain
from typing_extensions import List
from typing_extensions import Any
from config import UPLOAD_DIRECTORY
from utils.constants import Constants
from dependency_injector.wiring import inject
from fastapi import HTTPException, UploadFile, File
from utils.utils import embedding_llm_list, llm_list
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from utils.app_view_models import initialise_llm, initialise_embed_llm
from utils.app_view_models import (
    ingest_multi_doc,
    topics_from_pdf,
    topics_from_pdf_compare,
)

router = APIRouter()
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


## functions
def get_error_msg(err_msg, error):
    error_response = {
        "status": Constants.error,
        "data": {},
        "message": Constants.error_msg + f" {err_msg} " + str(error),
    }

    return error_response


## API
@router.get("/")
@inject
def home():
    return ""


@router.post("/download_model/")
@inject
async def download_model(request: dict) -> dict[str, Any]:
    """Prints the request body to the console."""
    input = request
    print(">>>>>>", input, type(input))
    model_name = input["model_name"]
    source = input["source"]

    if input["model_name"] and input["source"]:
        if source == "hf":
            try:
                embed_model = HuggingFaceEmbeddings(
                    model_name=model_name, show_progress=True
                )
                new_model = {
                    "visible_name": embed_model.model_name,
                    "info": {"domain": Domain.general.value},
                }

                embedding_llm_list.append(new_model)

                ## possibly need to store this in blob

                response_message = Constants.model_download_successful
                for model in embedding_llm_list:
                    if model["visible_name"] == new_model["visible_name"]:
                        response_message = f"Model already downloaded! Domain --> {model['info']['domain']}"
                        print(f"LOGG::: {response_message}")

                response = {
                    "status": Constants.success,
                    "data": new_model,
                    "message": response_message,
                }
                return response
            except Exception as e:
                print("An error occurred")
                response = {
                    "status": Constants.error,
                    "data": {},
                    "message": Constants.error_msg
                    + " Model download unsuccessful "
                    + str(e),
                }
                return response


@router.post("/save_documents/")
@inject
async def save_documents(files: List[UploadFile] = File(...)):
    """
    Saves uploaded documents to the server.
    """
    saved_files = []
    try:
        for file in files:
            unique_filename = file.filename

            file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)

            with open(file_path, "wb") as f:
                content = await file.read()  # Async read
                f.write(content)

            saved_files.append(
                {
                    "name": file.filename,
                    "type": file.content_type,
                    "size": len(content),
                    "path": file_path,  # Include the full path for easier access later
                }
            )

        return {"message": "Documents saved successfully", "files": saved_files}

    except Exception as e:  # Handle potential errors like file writing issues
        raise HTTPException(status_code=500, detail=f"Error saving files: {e}")


@router.post("/process_documents/")
def process_documents(request: dict):
    print("Input map >>")
    print(request)
    print("Input map >>")
    if request:
        try:
            ## get only the names of the uploaded files
            uploaded_files = request["uploaded_files"]
            embed_llm = [
                model
                for model in embedding_llm_list
                if model["visible_name"] == request["embed_llm"]
            ][0]
            llm = [
                model for model in llm_list if model["visible_name"] == request["llm"]
            ][0]
            embed_model_name = request["embed_model_name"]

            print("PROCESS DOCUMENTS >>>")
            print(uploaded_files)
            print(embed_llm)
            print(llm)
            print(embed_model_name)
            print("PROCESS DOCUMENTS >>>")

            ## initializing llm and embedding llm
            ## get the uploaded files list
            embed_llm = initialise_embed_llm(em_llm_opn=embed_llm)
            llm = initialise_llm(
                llm_source=llm["info"]["source"], llm_opn=llm["visible_name"]
            )

            insight_json_dict = ingest_multi_doc(
                uploaded_files, embed_llm, embed_model_name
            )

            print("Insight dict >>>>")
            print(insight_json_dict)

            key_insights_dict = {}
            if insight_json_dict:
                for key, value in insight_json_dict.items():
                    key_insights_dict[key] = topics_from_pdf(
                        llm=llm, list_topics=value, doc_name=key
                    )

            if len(insight_json_dict) >= 2:  # Ensure there are at least two documents
                # Get the first two items from the dictionary
                items = list(insight_json_dict.items())
                file1_name, file1_topics = items[0]
                file2_name, file2_topics = items[1]

                pdf_compare_insights = topics_from_pdf_compare(
                    list_of_topics_doc1=file1_topics,
                    list_of_topics_doc2=file2_topics,
                    document1=file1_name,
                    document2=file2_name,
                    llm=llm,
                )
            else:
                pdf_compare_insights = None  # Or handle the case where there are fewer than 2 docs appropriately

            print("Process documents results >>")
            print(
                insight_json_dict,
                key_insights_dict,
                pdf_compare_insights,
            )
            print("Process documents results >>")

            data = {
                "insight_json_dict": insight_json_dict,
                "key_insights_dict": key_insights_dict,
                "pdf_compare_insights": pdf_compare_insights,
            }

            ## Issue is that retriever dict contains objects and not json

            response = {
                "status": Constants.success,
                "data": data,
                "message": Constants.documents_processed_successful,
            }

            return response
        except Exception as e:
            return get_error_msg(err_msg="Document processing failed", error=e)
    else:
        return get_error_msg(err_msg="Input Empty", error="")
