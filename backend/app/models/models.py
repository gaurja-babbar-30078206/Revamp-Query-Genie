from enum import Enum


class Domain(Enum):
    openai = "Openai"
    general = "General"
    finance = "Financial"
    biomedical = "BioMedical"


## Model starts #####
from pydantic import BaseModel, Field, validator
from typing import List
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import PromptTemplate


class ThemeSubtheme(BaseModel):
    """Represents a theme and its associated subthemes."""

    theme: str = Field(description="The main theme extracted from the text.")
    subthemes: List[str] = Field(
        description="A list of subthemes related to the main theme."
    )


class DataFormat(BaseModel):
    """Data format for extracting themes and subthemes."""

    themes: List[ThemeSubtheme] = Field(
        description="A list of themes and their corresponding subthemes."
    )


# ---------------
class Themes(BaseModel):
    theme_name: str
    sub_themes: List[str]


class DataItem(BaseModel):
    common_theme_name: str
    common_sub_theme: List[str]
    json_1: Themes
    json_2: Themes


# --------------------------


class SubThemeDict(BaseModel):
    sub_themes: str = Field(
        description="The sub_theme within extracted from the theme."
    )
    description: str = Field(description="Description of the sub_theme")


class DataFormat(BaseModel):
    """Data format for extracting themes and subthemes."""

    themes: List[ThemeSubtheme] = Field(
        description="A list of themes and their corresponding subthemes."
    )


########### Comp App ########

# -------------------------------


class Subtheme(BaseModel):
    Theme: str
    Document_1: str
    Document_2: str


class Theme(BaseModel):
    theme_name: str
    sub_themes: List[Subtheme]


# --------------------------------


class ComparisonOutput(BaseModel):

    subtheme: str = Field(description="The subtheme being compared.")
    doc1_content: str = Field(
        description="Detailed content from the document related to the subtheme."
    )
    doc2_content: str = Field(
        description="Detailed content from the document related to the subtheme."
    )
    similarities: str = Field(
        description="Detailed similarities between the two documents regarding the subtheme."
    )
    differences: str = Field(
        description="Detailed differences between the two documents regarding the subtheme."
    )
    numerical_comparison: str = Field(
        description="Comparison of numerical data (if any), can be a markdown table or textual description."
    )


# ---------------------------------
from enum import Enum


### to view
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
