import os
import httpx
from fastapi import UploadFile, File
from utils.api_path import APIPaths
from utils.constants import blog
from typing import List
from langchain_community.vectorstores import FAISS

from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from typing import List
from config import VECTOR_STORE, UPLOAD_DIRECTORY

# upload, create vector stores, save their retrievers, create get their jsons
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    UnstructuredPowerPointLoader,
    PDFPlumberLoader,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)

import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    UnstructuredPowerPointLoader,
    PDFPlumberLoader,
)


# post
async def download_model(input_data: dict):
    """API to download the model, LLM or Embedding"""
    async with httpx.AsyncClient(
        timeout=6000
    ) as client:  # Use httpx.AsyncClient for asynchronous requests
        response = await client.post(
            url=APIPaths.download_model,
            json=input_data,  # Use json= for sending dictionaries as JSON
        )

    print(
        f"Download Model Status Code ::: {response.status_code}"
    )  # Check the status code
    print(f"Download Model Response :::{response.json()}")  # Parse the JSON response
    # return response


async def process_document(input_data: dict):
    """API to process documents"""
    async with httpx.AsyncClient(
        timeout=6000
    ) as client:  # Use httpx.AsyncClient for asynchronous requests
        response = await client.post(
            url=APIPaths.process_document,
            json=input_data,  # Use json= for sending dictionaries as JSON
        )

    print(
        f"Process Document Status Code ::: {response.status_code}"
    )  # Check the status code
    print(f"Process Document :::{response.json()}")  # Parse the JSON response
    return response.json()


async def save_documents(uploaded_files: List[UploadFile] = File(...)):
    """API to process documents"""

    if uploaded_files:
        files = []
        for uploaded_file in uploaded_files:
            files.append(
                (
                    "files",
                    (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type),
                )
            )

    async with httpx.AsyncClient(
        timeout=6000
    ) as client:  # Use httpx.AsyncClient for asynchronous requests
        response = await client.post(
            url=APIPaths.save_documents,
            files=files,  # Use json= for sending dictionaries as JSON
        )

    print(
        f"Save documents status code ::: {response.status_code}"
    )  # Check the status code
    print(f"Save documents response :::{response.json()}")  # Parse the JSON response
    # return response


async def get_theme_insights(input: dict):
    """Get theme insights"""
    with httpx.stream(
        method="GET", url=APIPaths.get_theme_insights, json=input, timeout=6000
    ) as r:
        for chunk in r.iter_raw():
            text_chunk = chunk.decode()  # Decode using default UTF-8
            yield text_chunk  # Yield the decoded string


async def get_comparison(input: dict):
    """Get comparison report"""
    async with httpx.AsyncClient(
        timeout=6000
    ) as client:  # Use httpx.AsyncClient for asynchronous requests
        response = await client.post(
            url=APIPaths.get_comparison,
            json=input,  # Use json= for sending dictionaries as JSON
        )

    print(f"Download Model Status Code ::: {response.status_code}")
    print(f"Download Model Response :::{response.json()}")
    return response.json()
