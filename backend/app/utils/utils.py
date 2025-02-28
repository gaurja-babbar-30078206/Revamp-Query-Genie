from models.models import LLMSource

embedding_llm_list = [
    {
        "visible_name": "text-embedding-ada-002",
        "info": {"domain": "openai"},
    },
    {"visible_name": "all-mpnet-base-v2", "info": {"domain": "general"}},
    {
        "visible_name": "paraphrase-MiniLM-L6-v2",
        "info": {"domain": "general"},
    },
    {
        "visible_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "info": {"domain": "general"},
    },
    {
        "visible_name": "FinLang/finance-embeddings-investopedia",
        "info": {"domain": "finance"},
    },
    {"visible_name": "ProsusAI/finbert", "domain": "finance"},
    {
        "visible_name": "emilyalsentzer/Bio_ClinicalBERT",
        "info": {"domain": "biomedical"},
    },
]

llm_list = [
    # OpenAI models
    {"visible_name": "gpt-4o-mini", "info": {"source": LLMSource.openai}},
    # Ollama models
    {"visible_name": "gemma2:9b", "info": {"source": LLMSource.ollama}},
    {"visible_name": "llama3.1:latest", "info": {"source": LLMSource.ollama}},
    {"visible_name": "llama3.2:3b", "info": {"source": LLMSource.ollama}},
    # Chatgroq models
    {"visible_name": "gemma2-9b-it", "info": {"source": LLMSource.chatgroq}},
]


response = {"status": "", "data": {}, "message": ""}
