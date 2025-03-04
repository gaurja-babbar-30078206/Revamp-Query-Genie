import sys
import asyncio
import pandas as pd

sys.path.append(".")
import streamlit as st
from models import DataType, LLMSource, Domain
from state import *
from router import (
    init,
    download_model,
    process_document,
    save_documents,
    get_theme_insights,
    get_comparison,
    get_chat_response,
)

st.set_page_config(page_title="Doc Insights & Comparision")

# Custom CSS for background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: lavender;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.sidebar.title("Options")

# init api which will fill the lists of llm and emb
if not st.session_state.init:
    init_response = asyncio.run(init())
    st.session_state.llm = init_response["llm"]
    st.session_state.embed_llm = init_response["embed_model"]
    st.session_state.em_llm_list = init_response["embeddings"]
    st.session_state.llm_list = init_response["models"]
    st.session_state.init = True

# Sidebar options (Data, Embedding, LLM, Document Upload)
with st.sidebar:
    st.session_state.data_type = st.selectbox(
        "Data", options=DataType, format_func=lambda x: x.value
    )

    if st.session_state.data_type == DataType.web:
        if url := st.text_input("Enter URL 👇"):
            st.session_state.web_url = url
            st.success("URL loaded!")

    st.session_state.domain = st.selectbox(
        label="Embedding Domain",
        placeholder="Choose a domain",
        options=Domain,  # Replace Domain with your actual domain options
        format_func=lambda x: x.value,
    )

    if st.session_state.domain:
        st.session_state.embed_llm_opn = st.selectbox(
            label="Embed LLM",
            placeholder="Choose an embed LLM",
            options=st.session_state.em_llm_list,
            format_func=lambda x: x["visible_name"],
            key="embed_llm_selectbox",  # Add a key for the selectbox
        )

        if st.session_state.embed_llm_opn:
            if (
                "embed_llm" not in st.session_state
                or "embed_model_name" not in st.session_state
                or st.session_state.embed_llm is None
                or st.session_state.embed_model_name is None
            ):
                st.session_state.embed_llm = st.session_state.embed_llm_opn[
                    "visible_name"
                ]
                st.session_state.embed_model_name = st.session_state.embed_llm_opn[
                    "visible_name"
                ]
                print("embed_model_option--->", st.session_state.embed_model_name)

    st.session_state.embed_model_input = st.text_input(
        "Download your own embedding model: 👇 "
    )
    if st.session_state.embed_model_input:
        if source := st.radio("Source", options=["hf", "🦙"], horizontal=True):
            # download_embed_model(model_name=st.session_state.embed_model_input, source=source)
            input = {"model_name": st.session_state.embed_model_input, "source": source}
            asyncio.run(download_model(input_data=input))

    st.session_state.llm_source = st.selectbox(
        label="LLM Source", options=LLMSource, format_func=lambda x: x.value
    )

    # if st.session_state.llm_source:
    st.session_state.llm_opn = st.selectbox(
        label="LLM",
        placeholder="Choose an LLM",
        options=st.session_state.llm_list,
        format_func=lambda x: x["visible_name"],
    )
    st.session_state.llm = st.session_state.llm_opn["visible_name"]

    st.session_state.uploaded_files = st.file_uploader(
        "Choose document(s)", accept_multiple_files=True, type=["pdf", "docx", "txt"]
    )

    if st.button("Process Documents"):
        if st.session_state.uploaded_files:
            with st.spinner("Processing documents..."):
                asyncio.run(
                    save_documents(uploaded_files=st.session_state.uploaded_files)
                )
                uploaded_files_name = [
                    file.name for file in st.session_state.uploaded_files
                ]

                input = {
                    "uploaded_files": uploaded_files_name,
                    "embed_llm": st.session_state.embed_llm,
                    "llm": st.session_state.llm,
                    "embed_model_name": st.session_state.embed_model_name,
                }

                response = asyncio.run(process_document(input_data=input))
                data = response["data"]
                st.session_state.insight_json_dict = data["insight_json_dict"]
                st.session_state.key_insights_dict = data["key_insights_dict"]
                st.session_state.pdf_compare_insights = data["pdf_compare_insights"]


tab1, tab2, tab3 = st.tabs(["Document Insights", "Document Comparison", "Chatbot"])

# Doc Insights Tab
with tab1:
    document_name = st.selectbox(
        "Select Document:", list(st.session_state.key_insights_dict.keys())
    )
    if document_name is not None:
        themes = [
            theme["theme"]
            for theme in st.session_state.key_insights_dict[document_name]["themes"]
        ]
        selected_theme = st.selectbox(
            "Select Theme:", themes, key="document_theme_selectbox"
        )  # Key added here

        # Subtheme dropdown
        if selected_theme:
            selected_theme_index = themes.index(selected_theme)
            subthemes = st.session_state.key_insights_dict[document_name]["themes"][
                selected_theme_index
            ]["subthemes"]
            selected_subtheme = st.selectbox(
                "Select Subtheme:", subthemes, key="document_subtheme_selectbox"
            )  # Key added here

        # Get Insights Button
        if selected_theme and selected_subtheme:
            if st.button("Get Insights"):
                st.session_state.get_insights = True

        if (
            selected_theme and selected_subtheme and st.session_state.get_insights
        ):  # Check get_insights flag
            with st.spinner("Generating insights..."):  # Added spinner
                uploaded_files_name = [
                    file.name for file in st.session_state.uploaded_files
                ]
                input = {
                    "file_list": uploaded_files_name,
                    "theme_name": selected_theme,
                    "subtheme_name": selected_subtheme,
                    "filename": document_name,
                    "embed_llm": st.session_state.embed_llm,
                    "llm": st.session_state.llm,
                }

                st.write(get_theme_insights(input=input))
            st.session_state.get_insights = False  # Reset the flag

# Doc Comparison Tab
with tab2:
    st.title("Document Comparison")

    if (
        st.session_state.insight_json_dict
        and len(list(st.session_state.insight_json_dict.keys())) >= 2
    ):
        themes = [
            theme["theme"] for theme in st.session_state.pdf_compare_insights["themes"]
        ]
        selected_theme = st.selectbox("Select Theme:", themes, key="theme_selectbox")

        if selected_theme:
            selected_theme_index = themes.index(selected_theme)
            subthemes = st.session_state.pdf_compare_insights["themes"][
                selected_theme_index
            ]["subthemes"]
            selected_subtheme = st.selectbox(
                "Select Subtheme:", subthemes, key="subtheme_selectbox"
            )

        if selected_theme and selected_subtheme:
            if st.button("Compare Documents"):
                st.session_state.compare_docs = True

        if selected_theme and selected_subtheme and st.session_state.compare_docs:
            with st.spinner("Comparing Documents..."):
                uploaded_files_name = [
                    file.name for file in st.session_state.uploaded_files
                ]
                input = {
                    "file_list": uploaded_files_name,
                    "theme_name": selected_theme,
                    "subtheme_name": selected_subtheme,
                    "embed_llm": st.session_state.embed_llm,
                    "llm": st.session_state.llm,
                }

                response = asyncio.run(get_comparison(input=input))
                print("FRONTEND RESPONSE >>")
                print(response)
                print("FRONTEND RESPONSE >>")

                data = response["data"]["data"]
                doc_1_name = response["data"]["doc_1_name"]
                doc_2_name = response["data"]["doc_2_name"]

            print("Data--->", data)
            df = pd.DataFrame(data["talking_points"])
            df.columns = ["Aspects", doc_1_name, doc_2_name]
            print("Extracted Data---->", data)
            if data:
                st.title(selected_subtheme.title())
                st.header("Document Comparison")
                st.write(df.to_markdown(index=False))
            st.session_state.compare_docs = False
    elif (
        st.session_state.insight_json_dict
        and len(st.session_state.insight_json_dict) < 2
    ):
        st.write("Please upload at least two documents for comparison")


# Chatbot Tab
with tab3:
    st.title("Ask Your Questions")
    uploaded_files_name = [file.name for file in st.session_state.uploaded_files]
    document_name = st.selectbox(
        "Select Document:",
        list(uploaded_files_name),
        key="Chatbot Document Selection",
    )
    if document_name is not None:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query := st.chat_input("Enter your query"):  # Get user input
            # Check if the query is not empty
            if query:  #  Crucial check!

                input = {
                    "file_list": uploaded_files_name,
                    "embed_llm": st.session_state.embed_llm,
                    "llm": st.session_state.llm,
                    "filename": document_name,
                    "query": query,
                }

                with st.chat_message("user"):
                    st.markdown(query)
                st.session_state.chat_history.append({"role": "user", "content": query})
                response = ""
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    st.write(get_chat_response(input=input))

                message_placeholder.markdown(response)
            else:  # Handle empty input
                st.warning("Please enter a query.")  # More user-friendly message
    else:
        st.write("Please upload and process documents first.")  # Keep this message
