from models.models import LLMSource, Domain

embedding_llm_list = [
    {
        "visible_name": "text-embedding-ada-002",
        "info": {"domain": Domain.openai.value},
    },
    {"visible_name": "all-mpnet-base-v2", "info": {"domain": Domain.general.value}},
    # {
    #     "visible_name": "paraphrase-MiniLM-L6-v2",
    #     "info": {"domain": Domain.general.value},
    # },
    # {
    #     "visible_name": "paraphrase-multilingual-MiniLM-L12-v2",
    #     "info": {"domain": Domain.general.value},
    # },
    {
        "visible_name": "FinLang/finance-embeddings-investopedia",
        "info": {"domain": Domain.finance.value},
    },
    # {"visible_name": "ProsusAI/finbert", "info": {"domain": Domain.finance.value}},
    {
        "visible_name": "emilyalsentzer/Bio_ClinicalBERT",
        "info": {"domain": Domain.biomedical.value},
    },
]

llm_list = [
    # OpenAI models
    {"visible_name": "gpt-4o-mini", "info": {"source": LLMSource.openai.value}},
    # Ollama models
    # {"visible_name": "gemma2:9b", "info": {"source": LLMSource.ollama.value}},
    # {"visible_name": "llama3.1:latest", "info": {"source": LLMSource.ollama.value}},
    # {"visible_name": "llama3.2:3b", "info": {"source": LLMSource.ollama.value}},
    # Chatgroq models
    {"visible_name": "gemma2-9b-it", "info": {"source": LLMSource.chatgroq.value}},
]
