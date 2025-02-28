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


def create_current_db(vec_path, embed_model, docs):
    if os.path.exists(vec_path):
        db = FAISS.load_local(
            vec_path, embed_model, allow_dangerous_deserialization=True
        )
    else:
        ## Created individual vector store
        db = FAISS.from_documents(documents=docs, embedding=embed_model)
        db.save_local(vec_path)

    return db


def return_multi_retriever(db, embed_model):
    retriever_sim = db.as_retriever(search_type="similarity", search_kwargs={"k": 15})
    retriever_mmr = db.as_retriever(search_type="mmr", search_kwargs={"k": 15})
    merger = MergerRetriever(retrievers=[retriever_sim, retriever_mmr])
    filter = EmbeddingsRedundantFilter(embeddings=embed_model)
    reordering = LongContextReorder()
    pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=merger
    )
    return compression_retriever


def return_documents(file_path):
    root, extension = os.path.splitext(file_path)

    if extension == ".pdf":
        documents = PDFPlumberLoader(file_path).load()
    elif extension == ".pptx":
        documents = UnstructuredPowerPointLoader(file_path).load()

    return documents


def preprocess(text, stop_words):
    """
    Tokenizes and preprocesses the input text, removing stopwords and short
    tokens.

    Parameters:
        text (str): The input text to preprocess.
        stop_words (set): A set of stopwords to be removed from the text.
    Returns:
        list: A list of preprocessed tokens.
    """
    result = []
    for token in simple_preprocess(text, deacc=True):
        if token not in stop_words and len(token) > 3:
            result.append(token)
    return result


def get_topic_lists_from_pdf(doc_lst, num_topics, words_per_topic):
    """
    Extracts topics and their associated words from a PDF document using the
    Latent Dirichlet Allocation (LDA) algorithm.

    Parameters:
        file (str): The path to the PDF file for topic extraction.
        num_topics (int): The number of topics to discover.
        words_per_topic (int): The number of words to include per topic.

    Returns:
        list: A list of num_topics sublists, each containing relevant words
        for a topic.
    """
    # Load the pdf file
    # loader = PdfReader(file)

    # Extract the text from each page into a list. Each page is considered a document
    documents = []
    for page in doc_lst:
        documents.append(page.page_content)

    # Preprocess the documents
    # nltk.download('stopwords')
    stop_words = set(stopwords.words(["english", "spanish"]))
    processed_documents = [preprocess(doc, stop_words) for doc in documents]

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(
        processed_documents, min_count=2, threshold=20
    )  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    bigram_docs = [bigram_mod[doc] for doc in processed_documents]

    # Create a dictionary and a corpus
    dictionary = corpora.Dictionary(bigram_docs)
    corpus = [dictionary.doc2bow(doc) for doc in bigram_docs]

    # Build the LDA model
    lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

    # Retrieve the topics and their corresponding words
    topics = lda_model.print_topics(num_words=30)

    # Store each list of words from each topic into a list
    topics_ls = []
    for topic in topics:
        words = topic[1].split("+")
        topic_words = [word.split("*")[1].replace('"', "").strip() for word in words]
        topics_ls.append(topic_words)

    return topics_ls


def ingest_multi_doc(file_list, embed_model, embed_model_name):
    retriever_dict = {}
    for uploaded_file in file_list:
        path = rf"{os.path.join(UPLOAD_DIRECTORY, uploaded_file)}"
        file_name = os.path.basename(path)
        file_name_without_ext = os.path.splitext(file_name)[
            0
        ]  # Get filename without extension
        vec_path = os.path.join(VECTOR_STORE, file_name, embed_model_name)

        ## creating jsons
        docs = return_documents(path)
        db = create_current_db(vec_path, embed_model, docs=docs)
        compression_retriever = return_multi_retriever(db=db, embed_model=embed_model)
        blog(f"Retriever created of ")
        blog(compression_retriever)
        retriever_dict[file_name] = compression_retriever  # Store in dictionary

    return retriever_dict
