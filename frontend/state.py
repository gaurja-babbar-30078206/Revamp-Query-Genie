import streamlit as st

# Initialize session state variables if they don't exist
if "init" not in st.session_state:
    st.session_state.init = False
if "insight_json_dict" not in st.session_state:
    st.session_state.insight_json_dict = {}
if "key_insights_dict" not in st.session_state:
    st.session_state.key_insights_dict = {}
if "pdf_compare_insights" not in st.session_state:
    st.session_state.pdf_compare_insights = {}
if "retriever_dict" not in st.session_state:
    st.session_state.retriever_dict = {}
if "selected_theme_ins" not in st.session_state:
    st.session_state.selected_theme_ins = None
if "selected_subtheme_ins" not in st.session_state:
    st.session_state.selected_subtheme_ins = None
if "get_insights" not in st.session_state:
    st.session_state.get_insights = False
if "compare_docs" not in st.session_state:
    st.session_state.compare_docs = False
if "final_comp_json" not in st.session_state:
    st.session_state.final_comp_json = None
if "embed_model_name" not in st.session_state:
    st.session_state.embed_model_name = None
if "domain" not in st.session_state:
    st.session_state.domain = None
if "embed_llm_opn" not in st.session_state:
    st.session_state.embed_llm_opn = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
if "em_llm_list" not in st.session_state:
    st.session_state.em_llm_list = None
if "llm_list" not in st.session_state:
    st.session_state.llm_list = None
if "llm" not in st.session_state:
    st.session_state.llm = None
