import os
import yaml

from langchain.chains import RetrievalQA
from fastapi import APIRouter
from models.models import Domain, ComparisonOutput1
from typing_extensions import List
from typing_extensions import Any
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

from utils.common_functions import get_gpt_mini
from langchain_core.prompts import PromptTemplate
from fastapi.responses import StreamingResponse
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
from langchain_core.output_parsers import PydanticOutputParser


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config_file_path = os.path.join(root_dir, "config.yml")

## loading config
with open(config_file_path, "r") as yamlfile:
    config = yaml.load(stream=yamlfile, Loader=yaml.Loader)


## load from config
index = os.path.join(root_dir, config["index"])
vector_store = os.path.join(root_dir, config["vector_store"])
upload_directory = os.path.join(root_dir, config["upload_directory"])

# creating required folders
os.makedirs(index, exist_ok=True)
os.makedirs(vector_store, exist_ok=True)
os.makedirs(upload_directory, exist_ok=True)

## variables
llm = None
embed_model = None
retriever_dict = {}

router = APIRouter()


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

            file_path = os.path.join(upload_directory, unique_filename)

            print("GAURJA >>")
            print(os.path.exists(file_path))
            print(upload_directory)
            print(unique_filename)
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
        # try:
        ## get only the names of the uploaded files
        uploaded_files = request["uploaded_files"]
        embed_model_name = request["embed_model_name"]
        embed_model = initialise_embed_llm(embed_llm=request["embed_llm"])
        llm = initialise_llm(llm_name=request["llm"])

        print("PROCESS DOCUMENTS >>>")
        print(uploaded_files)
        print(embed_model)
        print(llm)
        print(embed_model_name)
        print("PROCESS DOCUMENTS >>>")

        insight_json_dict = ingest_multi_doc(
            uploaded_files, embed_model, embed_model_name, upload_directory
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
    #     except Exception as e:
    #         return get_error_msg(err_msg="Document processing failed", error=e)
    # else:
    #     return get_error_msg(err_msg="Input Empty", error="")


## get-insights
def format_docs(docs, file_name=None):  # Add filename parameter
    return f"Information from '{file_name}':\n\n" + "\n\n".join(
        doc.page_content for doc in docs
    )


def get_rag_chain(llm):
    insight_temp = """
    <instruction>
    You are an expert business insights analyst.
    You will be provided with a theme and a sub-theme within that theme.
    Your task is to analyze the provided context and extract the most informative insights related to the given sub-theme.
    Your insights should be specific, quantifiable whenever possible. Prioritize insights that has more importance
    </instruction>

    <context>
    Context: {context}
    </context>

    <theme>
    Theme: {theme}
    </theme>

    <sub_theme>
    Sub-theme: {sub_themes}
    </sub_theme>

    Output:
    """
    prompt = PromptTemplate.from_template(insight_temp)
    rag_chain = prompt | llm
    return rag_chain


def return_documents(file_path):
    root, extension = os.path.splitext(file_path)

    if extension == ".pdf":
        documents = PDFPlumberLoader(file_path).load()
    elif extension == ".pptx":
        documents = UnstructuredPowerPointLoader(file_path).load()

    return documents


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


def get_retriever_dict(file_list, embed_model):
    for uploaded_file in file_list:
        path = rf"{os.path.join(upload_directory, uploaded_file)}"
        file_name = os.path.basename(path)
        file_name_without_ext = os.path.splitext(file_name)[
            0
        ]  # Get filename without extension
        vec_path = os.path.join(vector_store, file_name, embed_model.model)

        ## creating jsons
        docs = return_documents(path)
        db = create_current_db(vec_path, embed_model, docs=docs)
        compression_retriever = return_multi_retriever(db=db, embed_model=embed_model)
        print(f"Retriever created of ")
        print(compression_retriever)
        retriever_dict[file_name] = compression_retriever  # Store in dictionary
    return retriever_dict


def stream_insight(llm, retriever_dict, theme_name, subtheme_name, filename):
    ## getting retriever dict here
    theme = theme_name
    sub_theme = subtheme_name

    query = (
        f"In detail tell me about {theme}, and the sub-themes {', '.join(sub_theme)}"
    )

    if filename not in retriever_dict:
        raise ValueError(f"File '{filename}' not found in the retriever dictionary.")

    retriever = retriever_dict[filename]

    context = format_docs(retriever.invoke(query), file_name=filename)

    rag_chain = get_rag_chain(llm=llm)

    for chunk in rag_chain.stream(
        {"theme": theme, "sub_themes": sub_theme, "context": context}
    ):
        yield chunk.content


@router.get("/get_theme_insights/")
async def get_theme_insights(request: dict):

    embed_model = initialise_embed_llm(embed_llm=request["embed_llm"])
    llm = initialise_llm(llm_name=request["llm"])
    retriever_dict = get_retriever_dict(
        file_list=request["file_list"], embed_model=embed_model
    )

    return StreamingResponse(
        stream_insight(
            llm=llm,
            filename=request["filename"],
            retriever_dict=retriever_dict,
            theme_name=request["theme_name"],
            subtheme_name=request["subtheme_name"],
        ),
        media_type="text/event-stream",
    )


## get comparison


def extract_document_name(file_path):
    # Extract the base name of the file
    base_name = os.path.basename(file_path)
    # Split the base name to remove the extension
    document_name = os.path.splitext(base_name)[0]
    return document_name


def get_context_with_page_number(context):
    doc_name = extract_document_name(context[0].metadata["file_path"])
    new_context = ""
    new_context = f"""Context from {doc_name} is being given in chunks, each chunk of context consists of the page number from where it has been picked and the page content . Below is the context:"""

    for item in context:
        new_context += f"\n[The below context chunk is from page number: {int(item.metadata['page'])+1}]\n"
        new_context += item.page_content
    return new_context, doc_name


def get_context(retriever_dict, theme, sub_theme):
    query = (
        f"""Find out the relevant documents for the {sub_theme} within the {theme} """
    )

    file_name_1 = list(retriever_dict.keys())[0]  # Assuming you have at least two files
    file_name_2 = list(retriever_dict.keys())[1]

    retriever_1 = retriever_dict[file_name_1]
    retriever_2 = retriever_dict[file_name_2]

    context_1, doc_1_name = get_context_with_page_number(retriever_1.invoke(query))
    context_2, doc_2_name = get_context_with_page_number(retriever_2.invoke(query))

    return (context_1, doc_1_name, context_2, doc_2_name)


@router.post("/get_comparison/")
def get_detailed_comparison_chain1(request: dict):
    try:
        embed_model = initialise_embed_llm(embed_llm=request["embed_llm"])
        llm = initialise_llm(llm_name=request["llm"])
        retriever_dict = get_retriever_dict(
            file_list=request["file_list"], embed_model=embed_model
        )

        context_1, doc_1_name, context_2, doc_2_name = get_context(
            retriever_dict=retriever_dict,
            theme=request["theme_name"],
            sub_theme=request["subtheme_name"],
        )

        output_parser = PydanticOutputParser(pydantic_object=ComparisonOutput1)

        detailed_insight_prompt = PromptTemplate(
            template="""
            Compare two documents on a specific subtheme and extract comparable talking points.

            **Main Theme:** {theme}
            **Subtheme:** {sub_theme}

            **Document 1: {doc_1_name}**
            Content: {context_1}


            **Document 2: {doc_2_name}**
            Content: {context_2}


            **Instructions:**

            1. **Identify Talking Points:**  Find common themes or aspects discussed in both documents related to the subtheme. These will be your "talking points."

            2. **Extract Perspectives:** For each talking point, summarize the perspective or key information presented in *both* documents.

            3. **Structure Output:** Your output MUST adhere to the following JSON structure:

            {format_instructions}

            """,
            input_variables=[
                "theme",
                "sub_theme",
                "context_1",
                "context_2",
                "doc_1_name",
                "doc_2_name",
            ],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
        )

        comparison_chain = detailed_insight_prompt | llm | output_parser
        output = comparison_chain.invoke(
            {
                "theme": request["theme_name"],
                "sub_theme": request["subtheme_name"],
                "context_1": context_1,
                "context_2": context_2,
                "doc_1_name": doc_1_name,
                "doc_2_name": doc_2_name,
            }
        )
        data = {"data": output, "doc_1_name": doc_1_name, "doc_2_name": doc_2_name}

        response = {
            "status": Constants.success,
            "data": data,
            "message": Constants.document_comparison_successful,
        }

        return response
    except Exception as e:
        return get_error_msg(err_msg="Error in Document comparison", error=str(e))


## chat api
def create_chatbot_chain(retriever, llm, query):
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know.
    {context}
    Question: {question}"""
    QA_PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    context = format_docs(retriever.invoke(query))
    qa_chain = QA_PROMPT | llm
    for chunk in qa_chain.stream({"context": context, "question": query}):
        yield chunk.content


@router.get("/get_chat_response/")
async def get_chat_response(request: dict):
    embed_model = initialise_embed_llm(embed_llm=request["embed_llm"])
    llm = initialise_llm(llm_name=request["llm"])
    retriever_dict = get_retriever_dict(
        file_list=request["file_list"], embed_model=embed_model
    )
    retriever = retriever_dict[request["filename"]]
    return StreamingResponse(
        create_chatbot_chain(llm=llm, retriever=retriever, query=request["query"]),
        media_type="text/event-stream",
    )
