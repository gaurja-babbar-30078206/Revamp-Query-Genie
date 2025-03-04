from pathlib import Path
import streamlit as st
from utils.constants import blog
from utils.common_functions import get_gpt_mini, get_gpt_embedding
from config import INDEX, VECTOR_STORE, JSON_STORE


# App state for Insights
class InsigthAppState:
    def __init__(self):
        # self.init_folders()

        if "llm" not in st.session_state:
            st.session_state.llm = get_gpt_mini()
            blog("LLM initialized!")

        if "embed_llm" not in st.session_state:
            st.session_state.embed_llm = get_gpt_embedding()
            blog("Embed LLM initialized!")

        if "retriever_list" not in st.session_state:
            st.session_state.retriever_list = None
            blog("Embed LLM initialized!")

        if "insight_json_list" not in st.session_state:
            st.session_state.insight_json_list = None

        if "final_comp_json" not in st.session_state:
            st.session_state.final_comp_json = None

        if "selected_index" not in st.session_state:
            st.session_state.selected_index = None

        if "buttons_created" not in st.session_state:
            st.session_state["buttons_created"] = False

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "isChat" not in st.session_state:
            st.session_state.isChat = False

    # def init_folders(self):
    #     blog("Creating neccessary folders!")

    #     if not Path(INDEX).exists():
    #         Path.mkdir(INDEX)

    #     if not Path(VECTOR_STORE).exists():
    #         Path.mkdir(VECTOR_STORE)

    #     if not Path(JSON_STORE).exists():
    #         Path.mkdir(JSON_STORE)


class ComparisonAppState:

    def __init__(self):

        if "llm" not in st.session_state:
            st.session_state.llm = get_gpt_mini()
            blog("LLM initialized!")

        if "embed_llm" not in st.session_state:
            st.session_state.embed_llm = get_gpt_embedding()
            blog("Embed LLM initialized!")

        if "retriever_list" not in st.session_state:
            st.session_state.retriever_list = None
            blog("Embed LLM initialized!")

        if "insight_json_list" not in st.session_state:
            st.session_state.insight_json_list = None

        if "final_comp_json" not in st.session_state:
            st.session_state.final_comp_json = None

        if "selected_index_2" not in st.session_state:
            st.session_state.selected_index_2 = None

        if "isAutomaticInsights" not in st.session_state:
            st.session_state.isAutomaticInsights = None
