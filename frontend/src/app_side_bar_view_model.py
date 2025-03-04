import re
import os
import tempfile
import pandas as pd
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredPowerPointLoader, PDFPlumberLoader,UnstructuredURLLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.merger_retriever import MergerRetriever 
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
LongContextReorder)
from langchain_ollama import ChatOllama
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
os.environ["GROQ_API_KEY"] = "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"
from langchain_groq import ChatGroq

from utils.common_functions import get_gpt_mini,get_gpt_embedding
## Parameters
from app_models import Models, LLMSource, Domain
from enum import Enum
## Constants    

gen_embedding_models = [
        Models(visible_name= "all-mpnet-base-v2", info ={"size": ""} ),
        Models(visible_name= "paraphrase-MiniLM-L6-v2", info ={"size":""} ),
        Models(visible_name= "paraphrase-multilingual-MiniLM-L12-v2", info ={"size":""} ),
]
finance_embeddings = [
    Models(visible_name= "FinLang/finance-embeddings-investopedia", info ={"size": ""} ),
    Models(visible_name= "ProsusAI/finbert", info ={"size": ""} ),
]

biomedical_embeddings = [
    Models(visible_name= "emilyalsentzer/Bio_ClinicalBERT", info ={"size": ""} ),
]

legal_embeddings = []
clinical_embeddings = []
scientific_embeddings = []


domain_list = ["OpenAI","General", "Finance","Biomedical"]

llm_type_list = ["OpenAI","Ollama","Chatgroq"]



llm_list = [
    # OpenAI models
    Models(visible_name= "gpt-4o-mini" , info= {"source": LLMSource.openai } ),
    
    # Ollama models
    Models(visible_name= "gemma2:9b" , info= {"source": LLMSource.ollama } ),
    Models(visible_name= "llama3.1:latest" , info= {"source": LLMSource.ollama } ),
    Models(visible_name= "llama3.2:3b" , info= {"source": LLMSource.ollama } ),
    
    # Chatgroq models
    Models(visible_name= "gemma2-9b-it" , info= {"source": LLMSource.chatgroq } ),
]

embedding_llm_list = [

    ## General Purpose
    Models(visible_name= "text-embedding-ada-002", info ={"size": "","domain":Domain.openai} ),
    ## General Purpose
    Models(visible_name= "all-mpnet-base-v2", info ={"size": "","domain":Domain.general} ),
    Models(visible_name= "paraphrase-MiniLM-L6-v2", info ={"size":"","domain":Domain.general} ),
    Models(visible_name= "paraphrase-multilingual-MiniLM-L12-v2", info ={"size":"","domain":Domain.general} ),

    ## Finance
    Models(visible_name= "FinLang/finance-embeddings-investopedia", info ={"size": "","domain":Domain.finance} ),
    Models(visible_name= "ProsusAI/finbert", info ={"size": "","domain":Domain.finance}),
    
    ## Medical
    Models(visible_name= "emilyalsentzer/Bio_ClinicalBERT", info ={"size": "","domain":Domain.biomedical} ),
]
    


## Methods
def embed_hash_func(obj: Models):
    return obj.visible_name
    
# @st.cache_resource(hash_funcs= {Models : embed_hash_func})
# def initialise_embed_llm(embed_llm_opn):
#     if embed_llm_opn is not None:
#         print(f"llm embed initialising ... {embed_llm_opn}")
#         embed_llm = HuggingFaceEmbeddings(model_name = embed_llm_opn.visible_name,show_progress = True)  
#         return embed_llm 
  
def download_embed_model(model_name, source):
    if model_name and source:
        if source == "🤗":
            try:
                embed_model = HuggingFaceEmbeddings(model_name = model_name,show_progress = True)
                new_model = Models(visible_name=embed_model.model_name,info= {"domain": Domain.general})
                for model in embedding_llm_list:
                    if model.visible_name == new_model.visible_name:
                        st.info(f"Model already downloaded! Domain --> {model.info['domain'].value}")
                        return embed_model
                
                embedding_llm_list.append(new_model)
                st.success("Model downloaded successfully!✅")
                return embed_model
            except:
                print("An error occurred")
                st.error("Model not found", icon="🚨")




def hf_embed(obj: HuggingFaceEmbeddings):
    return obj.model_name

from langchain_core.documents import Document
def doc_hash(obj: list[Document]):
    return obj
    
def return_documents(file_path):
    root, extension = os.path.splitext(file_path)
    
    if (extension == ".pdf"):
        docs =  PDFPlumberLoader(file_path).load()
    elif (extension == ".pptx"):
        docs =  UnstructuredPowerPointLoader(file_path).load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    documents = splitter.split_documents(docs)
    return documents     

 
def ingest_doc(uploaded_file, embed_model):
    temp_dir = tempfile.mkdtemp()
    path = rf"{os.path.join(temp_dir,uploaded_file.name)}"
    with open(path,"wb") as f:
        f.write(uploaded_file.getvalue())
    file_name = os.path.basename(path)
    
    print(f"Path: {path}, Uploaded file: {uploaded_file}")
    docs =  return_documents(path)
    compression_retriever = return_retriever(docs, embed_model)
    return compression_retriever
                             
# @st.cache_resource(hash_funcs= {HuggingFaceEmbeddings: hf_embed, list[Document]: doc_hash})   
def return_retriever(docs, embed_model):
    db = FAISS.from_documents(documents= docs, embedding = embed_model)
    retriever_sim = db.as_retriever(search_type="similarity", search_kwargs={"k":5 })
    retriever_mmr = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})   
    merger = MergerRetriever(retrievers=[retriever_sim, retriever_mmr])
    filter = EmbeddingsRedundantFilter(embeddings=embed_model)
    reordering = LongContextReorder()
    pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=merger)
    return compression_retriever


def ingest_web_url(url:str, embed_model):
    print(f"Ingesting url...")
    url_list = []
    url_list.append(url)
    loader = UnstructuredURLLoader(urls=url_list,show_progress_bar=True)
    print(f"URL --------> {url_list}")
    docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100))
    print(f"Web docs --> {docs}")
    compression_retriever = return_retriever(docs, embed_model)
    return compression_retriever            




def blog(input):
    print("\n")
    print(f"{datetime.now()}✨ ✨ ✨ -----> 🧮 🧮 🧮 {input} ")
    print("\n")


def remove_special_characters(input_string):
    return re.sub(r'[^a-zA-Z0-9\s]', '', input_string)

def upload_csv_files(file_list) -> str:
    # creating db at path
    # db_path = Path.joinpath(Path.cwd(),"sql_database.db")
    db_path = r"D:\OneDrive - Adani\Desktop\Final_Projects\QueryGenie\single_doc_query\single_app_doc\sql_data.db"
    print(f"DB path -------< {db_path}")
    db_uri = f"sqlite:///{db_path}"
    engine = create_engine(db_uri)
    db = SQLDatabase(engine= engine)
    
    for file in file_list:
        print(file)
        temp_dir = tempfile.mkdtemp()
        file_path = rf"{os.path.join(temp_dir,file.name)}"
        with open(file_path,"wb") as f:
            f.write(file.getvalue())

        table_name = remove_special_characters(os.path.splitext(file.name)[0])
        blog(f"Temp Final path --> {file_path}")
        blog(f"Table name ---> {table_name}")

        df = pd.read_csv(file_path)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
    
    print(db.get_usable_table_names())
    return db_uri    
        
    

def get_llm_list(llm_source: LLMSource):
    ## fill the list according to the llm source
    llm_result = [llm for llm in llm_list if llm.info["source"] == llm_source]
    return llm_result

def get_embed_llm_list(em_llm_source: Domain):
    ## fill the list according to the llm source
    llm_result = [em_llm for em_llm in embedding_llm_list if em_llm.info["domain"]== em_llm_source]
    return llm_result

@st.cache_resource(hash_funcs= {Models:embed_hash_func})  
def initialise_llm(llm_source: LLMSource, llm_opn: Models):
    print(f"Model name ---> {llm_opn}")
    model =  llm_opn.visible_name
    if llm_source == LLMSource.openai:
        return get_gpt_mini()
    elif llm_source == LLMSource.ollama:
        return ChatOllama(model= model,temperature=0)
    elif llm_source == LLMSource.chatgroq:
        return ChatGroq(
            model=model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
 
 ### Embedding llm

# @st.cache_resource(hash_funcs= {Models:embed_hash_func})       
def initialise_embed_llm(em_llm_opn:Models):
    print(f"Model name ---> {em_llm_opn}")   
    if em_llm_opn =="text-embedding-ada-002":
        return get_gpt_embedding(),em_llm_opn 
  
    else:
        return HuggingFaceEmbeddings(model_name = em_llm_opn,
                                    show_progress = True),em_llm_opn 
            


# # @st.cache_resource  # Use st.cache_resource (no hash_funcs needed for Models/Enums)
# # def initialise_embed_llm(em_llm_opn: Models):
# #     print(f"Model name ---> {em_llm_opn.visible_name}")
# #     if em_llm_opn.visible_name == "text-embedding-ada-002":
# #         return get_gpt_embedding(), em_llm_opn.visible_name
# #     else:
# #         return HuggingFaceEmbeddings(model_name=em_llm_opn.visible_name, show_progress=True), em_llm_opn.visible_name
    

# # @st.cache_resource
# # def initialise_embed_llm(model_name: str):
# #     print(f"Model name ---> {model_name}") # Debugging output
# #     if model_name == "text-embedding-ada-002":
# #         return get_gpt_embedding(), model_name
# #     else:
# #         return HuggingFaceEmbeddings(model_name=model_name, show_progress=True), model_name

# @st.cache_resource
# def initialise_embed_llm(model_name: str):
#     print(f"Model name ---> {model_name}")  # Debugging output
#     if model_name == "text-embedding-ada-002":
#         return get_gpt_embedding(), model_name
#     else:
#         return HuggingFaceEmbeddings(model_name=model_name, show_progress=True), model_name


# def hash_models(models):
#     return hash((models.visible_name, frozenset(models.info.items())))


# @st.cache_resource(hash_funcs={Models: hash_models})
# def initialise_llm(llm_source: LLMSource, llm_opn: Models):
#     print(f"Model name ---> {llm_opn.visible_name}")
#     model = llm_opn.visible_name
#     if llm_source == LLMSource.openai:
#         return get_gpt_mini()
#     elif llm_source == LLMSource.ollama:
#         return ChatOllama(model=model, temperature=0)
#     elif llm_source == LLMSource.chatgroq:
#         return ChatGroq(model=model, temperature=0, max_tokens=None, timeout=None, max_retries=2)

def create_current_vec_store(file_list: list[str], embed_model, index_path):
    
    dimensions: int = len(embed_model.embed_query("dummy"))
    db_current = FAISS(
    embedding_function= embed_model,
    index=IndexFlatL2(dimensions),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
    normalize_L2=False
    )
    
    print(f"Current db before adding ---> {db_current}")
    
    for file in file_list:
        file_name = os.path.basename(file)
        vec_path = os.path.join(index_path, file_name)
        
        if os.path.exists(vec_path):
            db = FAISS.load_local(vec_path, embed_model,allow_dangerous_deserialization = True)
            try:
                db_current.merge_from(db)
            except:
                print(f"Already exists --> {file}")
        else:
            ## Created individual vector store
            docs = PDFPlumberLoader(file).load_and_split(RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100))
            db =  FAISS.from_documents(documents=docs,embedding=embed_model)
            db.save_local(vec_path)
            ## Add it to the current vector store
            try:
                db_current.merge_from(db)
            except:
                print(f"Already exists --> {file}")     
    
    return db_current 
        
        
def ingest_multi_doc(file_list , embed_model):
    
    index_path = r"D:\OneDrive - Adani\Desktop\Final_Projects\QueryGenie\storage"
    
    print(f"File list --> {file_list}")
    dimensions: int = len(embed_model.embed_query("dummy"))
    db_current = FAISS(
    embedding_function= embed_model,
    index=IndexFlatL2(dimensions),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
    normalize_L2=False
    )
    
    for uploaded_file in file_list:       
        temp_dir = tempfile.mkdtemp()
        path = rf"{os.path.join(temp_dir,uploaded_file.name)}"
        with open(path,"wb") as f:
            f.write(uploaded_file.getvalue())
        file_name = os.path.basename(path)
        vec_path = os.path.join(index_path, file_name)
        db_current = create_current_db(vec_path,embed_model,db_current,path)
    
    compression_retriever = return_multi_retriever(db=db_current,embed_model=embed_model)
    return compression_retriever    
        
    
    
def create_current_db(vec_path, embed_model, db_current,path):
    if os.path.exists(vec_path):
            db = FAISS.load_local(vec_path, embed_model,allow_dangerous_deserialization = True)
            try:
                db_current.merge_from(db)
            except:
                print(f"Already exists --> {vec_path}")
    else:
        ## Created individual vector store
        docs =  return_documents(path)
        # docs = PDFPlumberLoader(vec_path).load_and_split(RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100))
        db =  FAISS.from_documents(documents=docs,embedding=embed_model)
        db.save_local(vec_path)
        ## Add it to the current vector store
        try:
            db_current.merge_from(db)
        except:
            print(f"Already exists --> {vec_path}")
    
    return db_current                


def return_multi_retriever(db, embed_model):
    retriever_sim = db.as_retriever(search_type="similarity", search_kwargs={"k":5 })
    retriever_mmr = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})   
    merger = MergerRetriever(retrievers=[retriever_sim, retriever_mmr])
    filter = EmbeddingsRedundantFilter(embeddings=embed_model)
    reordering = LongContextReorder()
    pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=merger)
    return compression_retriever


