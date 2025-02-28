import sys
import json
import asyncio

sys.path.append(".")
import streamlit as st
from app_models import DataType, LLMSource, Domain
from app_side_bar_view_model import (
    get_embed_llm_list,
    initialise_embed_llm,
    get_llm_list,
    initialise_llm,
)
from app_view_models import (
    ingest_multi_doc,
    get_theme_insight,
    create_chatbot_chain,
    get_detailed_comparison_chain1,
)
import pandas as pd
from api_service import (
    download_model,
    process_document,
    save_documents,
    ingest_multi_doc,
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

# Initialize session state variables if they don't exist
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


st.sidebar.title("Options")
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
        em_llm_list = get_embed_llm_list(em_llm_source=st.session_state.domain)
        st.session_state.embed_llm_opn = st.selectbox(
            label="Embed LLM",
            placeholder="Choose an embed LLM",
            options=em_llm_list,
            format_func=lambda x: x.visible_name,
            key="embed_llm_selectbox",  # Add a key for the selectbox
        )

        if st.session_state.embed_llm_opn:
            if (
                "embed_llm" not in st.session_state
                or "embed_model_name" not in st.session_state
                or st.session_state.embed_llm is None
                or st.session_state.embed_model_name is None
            ):
                st.session_state.embed_llm, st.session_state.embed_model_name = (
                    initialise_embed_llm(
                        em_llm_opn=st.session_state.embed_llm_opn.visible_name
                    )
                )
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

    if st.session_state.llm_source:

        llm_list = get_llm_list(st.session_state.llm_source)
    st.session_state.llm_opn = st.selectbox(
        label="LLM",
        placeholder="Choose an LLM",
        options=llm_list,
        format_func=lambda x: x.visible_name,
    )
    st.session_state.llm = initialise_llm(
        llm_source=st.session_state.llm_source, llm_opn=st.session_state.llm_opn
    )
    uploaded_files = st.file_uploader(
        "Choose document(s)", accept_multiple_files=True, type=["pdf", "docx", "txt"]
    )

    # uploaded_files = st.file_uploader(...)
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):  # Add spinner here

                # api to save documents
                asyncio.run(save_documents(uploaded_files=uploaded_files))
                uploaded_files_name = [file.name for file in uploaded_files]

                input = {
                    "uploaded_files": uploaded_files_name,
                    "embed_llm": st.session_state.embed_llm.model,
                    "llm": st.session_state.llm.model_name,
                    "embed_model_name": st.session_state.embed_model_name,
                }
                print(f"Uploaded files >> {uploaded_files_name}")
                print(f"LLM LLM >> {st.session_state.llm.model_name}")
                print(f"Embedding LLM >> {st.session_state.embed_llm.model}")

                response = asyncio.run(process_document(input_data=input))
                data = response["data"]

                st.session_state.insight_json_dict = data["insight_json_dict"]
                st.session_state.key_insights_dict = data["key_insights_dict"]
                st.session_state.pdf_compare_insights = data["pdf_compare_insights"]

                # ingest
                st.session_state.retriever_dict = ingest_multi_doc(
                    file_list=uploaded_files_name,
                    embed_model=st.session_state.embed_llm,
                    embed_model_name=st.session_state.embed_model_name,
                )

                print("GOT RETRIEVER DICT >>")
                print(st.session_state.retriever_dict)
                print("GOT RETRIEVER DICT >>")
                print("HELLO HELLO HELLO")


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
                st.write(
                    get_theme_insight(
                        llm=st.session_state.llm,
                        retriever_dict=st.session_state.retriever_dict,
                        theme_name=selected_theme,
                        subtheme_name=selected_subtheme,
                        filename=document_name,
                    )
                )
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
                data, doc_1_name, doc_2_name = get_detailed_comparison_chain1(
                    llm=st.session_state.llm,
                    theme=selected_theme,
                    sub_theme=selected_subtheme,
                    retriever_dict=st.session_state.retriever_dict,
                )
            print("Data--->", data)
            df = pd.DataFrame(data.talking_points)
            df.columns = ["Aspects", doc_1_name, doc_2_name]
            # df[doc_1_name] = df[doc_1_name].apply(lambda x: x['perspective'])
            # df[doc_2_name] = df[doc_2_name].apply(lambda x: x['perspective'])
            print("Extracted Data---->", data)
            if data:
                st.title(selected_subtheme.title())

                st.header("Document Comparison")
                st.write(df.to_markdown(index=False))
                # col1, col2 = st.columns(2)
            st.session_state.compare_docs = False
    elif (
        st.session_state.insight_json_dict
        and len(st.session_state.insight_json_dict) < 2
    ):
        st.write("Please upload at least two documents for comparison")


# Chatbot Tab
with tab3:
    st.title("Ask Your Questions")

    if st.session_state.retriever_dict:  # Check if retrievers are available
        document_name = st.selectbox(
            "Select Document:",
            list(st.session_state.retriever_dict.keys()),
            key="Chatbot Document Selection",
        )
        if document_name is not None:
            # Merge all retrievers into one for the chatbot
            retriever = st.session_state.retriever_dict[document_name]

            st.session_state.chatbot_chain = create_chatbot_chain(
                retriever, st.session_state.llm
            )

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if query := st.chat_input("Enter your query"):  # Get user input
                # Check if the query is not empty
                if query:  #  Crucial check!
                    with st.chat_message("user"):
                        st.markdown(query)
                    st.session_state.chat_history.append(
                        {"role": "user", "content": query}
                    )

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = ""

                        # Use the correct input key "query" (or adjust your chain to accept "question")
                        for response in st.session_state.chatbot_chain.stream(
                            {"query": query}
                        ):
                            # print("Response--------->",response,"<---------")
                            full_response += response["result"]
                            message_placeholder.markdown(full_response + " ")
                        message_placeholder.markdown(full_response)

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": full_response}
                    )
                else:  # Handle empty input
                    st.warning("Please enter a query.")  # More user-friendly message
        else:
            st.write("Please upload and process documents first.")  # Keep this message
