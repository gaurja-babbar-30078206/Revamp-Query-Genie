import os
import yaml
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config_file_path = os.path.join(root_dir, "config.yml")

## loading config
with open(config_file_path, "r") as yamlfile:
    config = yaml.load(stream=yamlfile, Loader=yaml.Loader)

print("CONFIG KEYS >>")
print(config.keys())


load_dotenv(find_dotenv())

AZURE_MODEL_DEPLOYMENT_ID = os.environ["AZURE_MODEL_DEPLOYMENT_ID"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_API_VERSION = os.environ["AZURE_API_VERSION"]
AZURE_MODEL_NAME = os.environ["AZURE_MODEL_NAME"]
CONTEXT_LENGTH = 10000


# llm
def get_gpt_mini():
    # Initialize LLM and embedding model
    llm = AzureChatOpenAI(
        model=AZURE_MODEL_NAME,
        deployment_name=AZURE_MODEL_DEPLOYMENT_ID,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )
    return llm


# embed llm
### Embedding llm
def get_gpt_embedding():
    from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings

    embeddings = AzureOpenAIEmbeddings(
        openai_api_key="07a2db3305d14f619202c549ca81b0d2",
        azure_endpoint="https://ailabazopenaise.openai.azure.com",
        deployment="ailabtxtembada",
        model="text-embedding-ada-002",
        chunk_size=4000,
        retry_min_seconds=15,
        retry_max_seconds=60,
        max_retries=20,
        show_progress_bar=True,
    )
    return embeddings


def init_folders():
    """Creation of required folders"""
    print("Creating neccessary folders!")
    index = config["index"]
    vector_store = config["vector_store"]
    json_store = config["json_store"]
    upload_directory = config["upload_directory"]

    if not os.path.exists(index):
        os.mkdir(index)

    if not os.path.exists(vector_store):
        os.mkdir(vector_store)

    if not os.path.exists(json_store):
        os.mkdir(json_store)

    if not os.path.exists(upload_directory):
        os.mkdir(upload_directory)
