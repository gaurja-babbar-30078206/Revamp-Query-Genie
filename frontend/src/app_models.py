## Model starts #####
from typing import List
from enum import Enum


class Models:
    visible_name = ""
    info = {}

    def __init__(self, visible_name=None, info=None) -> None:
        self.visible_name = visible_name
        self.info = info


# class Models:
#     def __init__(self, visible_name=None, info=None):
#         self.visible_name = visible_name
#         self.info = info

#     def __hash__(self):
#         return hash((self.visible_name, frozenset(self.info.items()) if self.info else None))

#     def __eq__(self, other):
#         return (self.visible_name, self.info) == (other.visible_name, other.info)


## Models
class LLM(Enum):
    gpt = "gpt"
    gemma = "gemma2:9b"
    llama = "llama3.1:8b"
    llama2 = "llama2:13b"
    gro_gemma = "gemma2-9b-it"


class EmbedLLM(Enum):
    finance_embed = "FinLang/finance-embeddings-investopedia"
    hf_embedding = "all-MiniLM-L6-v2"


class DataType(Enum):
    unstructured = "Textual"
    structured = "Tabular"
    web = "Web"


class LLMSource(Enum):
    openai = "Openai"
    ollama = "Ollama"
    chatgroq = "Chatgroq"


class Domain(Enum):
    openai = "Openai"
    general = "General"
    finance = "Financial"
    biomedical = "BioMedical"
