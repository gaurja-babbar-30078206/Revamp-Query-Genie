from langchain_openai import AzureChatOpenAI

AZURE_MODEL_DEPLOYMENT_ID = "agelgpt4o"
AZURE_OPENAI_API_KEY = "05bb0956ad57473d95900e3981288554"
AZURE_OPENAI_EMBED_API_KEY = "07a2db3305d14f619202c549ca81b0d2"
AZURE_OPENAI_ENDPOINT = "https://ailabazopenaius.openai.azure.com/"
AZURE_API_VERSION = "2024-02-15-preview"
AZURE_MODEL_NAME = "gpt-4o-mini"
AZURE_MODEL_DEPLOYMENT_ID = "agelgpt4o"
AZURE_EMBEDDING_DEPLOYMENT_ID = "ailabtxtembada"


# llm
def get_gpt_mini():

    AZURE_MODEL_DEPLOYMENT_ID = "agelgpt4o"
    AZURE_OPENAI_API_KEY = "05bb0956ad57473d95900e3981288554"
    AZURE_OPENAI_EMBED_API_KEY = "07a2db3305d14f619202c549ca81b0d2"
    AZURE_OPENAI_ENDPOINT = "https://ailabazopenaius.openai.azure.com/"
    AZURE_API_VERSION = "2024-02-15-preview"
    AZURE_MODEL_NAME = "gpt-4o-mini"
    AZURE_MODEL_DEPLOYMENT_ID = "agelgpt4o"
    AZURE_EMBEDDING_DEPLOYMENT_ID = "ailabtxtembada"
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
