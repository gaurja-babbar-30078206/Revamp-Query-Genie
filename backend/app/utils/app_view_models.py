from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import pandas as pd
import streamlit as st
from typing import List
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from models.models import DataFormat, DataItem, Theme, ComparisonOutput

# upload, create vector stores, save their retrievers, create get their jsons
import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    UnstructuredPowerPointLoader,
    PDFPlumberLoader,
    UnstructuredURLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain_core.runnables import RunnablePassthrough


from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
    LongContextReorder,
)

import gensim
import nltk
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from pypdf import PdfReader
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.llms import OpenAI
import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    UnstructuredPowerPointLoader,
    PDFPlumberLoader,
    UnstructuredURLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import UPLOAD_DIRECTORY

from utils.common_functions import get_gpt_mini, get_gpt_embedding
from langchain_huggingface import HuggingFaceEmbeddings
from models.models import LLMSource


def blog(input):
    print("🎃🎃🎃*LOGS*🎃🎃🎃")
    print(input)


######################### Documents Related ##################################
def return_documents(file_path):
    root, extension = os.path.splitext(file_path)

    if extension == ".pdf":
        documents = PDFPlumberLoader(file_path).load()
    elif extension == ".pptx":
        documents = UnstructuredPowerPointLoader(file_path).load()

    return documents


def format_docs(docs, file_name):  # Add filename parameter
    return f"Information from '{file_name}':\n\n" + "\n\n".join(
        doc.page_content for doc in docs
    )


def extract_document_name(file_path):
    # Extract the base name of the file
    base_name = os.path.basename(file_path)
    # Split the base name to remove the extension
    document_name = os.path.splitext(base_name)[0]
    return document_name


def create_current_db(vec_path, embed_model, docs):
    if os.path.exists(vec_path):
        db = FAISS.load_local(
            vec_path, embed_model, allow_dangerous_deserialization=True
        )
    else:
        ## Created individual vector store
        db = FAISS.from_documents(documents=docs, embedding=embed_model)
        db.save_local(vec_path)

    return db


def return_multi_retriever(db, embed_model):
    retriever_sim = db.as_retriever(search_type="similarity", search_kwargs={"k": 15})
    retriever_mmr = db.as_retriever(search_type="mmr", search_kwargs={"k": 15})
    merger = MergerRetriever(retrievers=[retriever_sim, retriever_mmr])
    filter = EmbeddingsRedundantFilter(embeddings=embed_model)
    reordering = LongContextReorder()
    pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=merger
    )
    return compression_retriever


def ingest_multi_doc(file_list, embed_model, embed_model_name):
    insight_json_dict = {}

    print("File list --> ")

    for uploaded_file in file_list:
        path = rf"{os.path.join(UPLOAD_DIRECTORY, uploaded_file)}"
        file_name = os.path.basename(path)
        file_name_without_ext = os.path.splitext(file_name)[
            0
        ]  # Get filename without extension

        ## creating jsons
        docs = return_documents(path)

        insight_json = get_topic_lists_from_pdf(
            doc_lst=docs, num_topics=10, words_per_topic=100
        )
        print("Generated_Insights--->")
        print(insight_json)
        print("END")
        insight_json_dict[file_name] = insight_json  # Store in dictionary
        blog(f"Json created of ---> ")
        blog(insight_json_dict)

        print("EMBED MODEL >>>")
        print(embed_model)
        print("EMBED MODEL >>>")
    return insight_json_dict


### Topic Modelling ###############


def preprocess(text, stop_words):
    """
    Tokenizes and preprocesses the input text, removing stopwords and short
    tokens.

    Parameters:
        text (str): The input text to preprocess.
        stop_words (set): A set of stopwords to be removed from the text.
    Returns:
        list: A list of preprocessed tokens.
    """
    result = []
    for token in simple_preprocess(text, deacc=True):
        if token not in stop_words and len(token) > 3:
            result.append(token)
    return result


def get_topic_lists_from_pdf(doc_lst, num_topics, words_per_topic):
    """
    Extracts topics and their associated words from a PDF document using the
    Latent Dirichlet Allocation (LDA) algorithm.

    Parameters:
        file (str): The path to the PDF file for topic extraction.
        num_topics (int): The number of topics to discover.
        words_per_topic (int): The number of words to include per topic.

    Returns:
        list: A list of num_topics sublists, each containing relevant words
        for a topic.
    """
    # Load the pdf file
    # loader = PdfReader(file)

    # Extract the text from each page into a list. Each page is considered a document
    documents = []
    for page in doc_lst:
        documents.append(page.page_content)

    # Preprocess the documents
    # nltk.download('stopwords')
    stop_words = set(stopwords.words(["english", "spanish"]))
    processed_documents = [preprocess(doc, stop_words) for doc in documents]

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(
        processed_documents, min_count=2, threshold=20
    )  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    bigram_docs = [bigram_mod[doc] for doc in processed_documents]

    # Create a dictionary and a corpus
    dictionary = corpora.Dictionary(bigram_docs)
    corpus = [dictionary.doc2bow(doc) for doc in bigram_docs]

    # Build the LDA model
    lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

    # Retrieve the topics and their corresponding words
    topics = lda_model.print_topics(num_words=30)

    # Store each list of words from each topic into a list
    topics_ls = []
    for topic in topics:
        words = topic[1].split("+")
        topic_words = [word.split("*")[1].replace('"', "").strip() for word in words]
        topics_ls.append(topic_words)

    return topics_ls


def topics_from_pdf(llm, list_topics, doc_name, max_retries=2):
    """
    Generates descriptive prompts for LLM based on topic words extracted from a PDF,
    combining similar topics and handling parsing errors with retries.
    """

    # Format topic lists into strings for the prompt
    def format_topics(topics, doc_num):
        topics_string = f"Document {doc_num} Topics:\n"
        for i, lst in enumerate(topics):
            topics_string += f"Topic {i+1}: {', '.join(lst)}\n"
        return topics_string

    topics_string = format_topics(list_topics, doc_name)

    parser = JsonOutputParser(pydantic_object=DataFormat)

    template_string = """
    Analyze the following lists of keywords, each representing a potential topic extracted from a document. 
    Your task is to describe each topic concisely, combine similar topics into broader themes, and provide relevant subthemes.
    Do not use single words or words connected by underscores as subthemes. Ensure the subthemes provide specific insights related to the theme.

    Keyword Lists:
    ```
    {topics_string}
    ```

    Output Format:  Provide your response in JSON format.

    If topics are very similar, combine them into a single theme with multiple subthemes representing the nuances of the combined topics.
    Ensure the "themes" list in the JSON output reflects these combined themes.  Do not create unnecessary subthemes. Aim for clear and distinct subthemes under each theme.

    JSON Output:
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template_string,
        input_variables=["topics_string"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    for attempt in range(max_retries):
        try:
            output = chain.invoke({"topics_string": topics_string})
            return output  # Returns the parsed DataFormat object
        except Exception as e:
            print(f"Error parsing LLM output on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                print(f"Failed to parse output after {max_retries} attempts.")
                return None  # Or handle the error as needed


def topics_from_pdf_compare(
    llm, list_of_topics_doc1, list_of_topics_doc2, document1, document2, max_retries=2
):
    """
    Generates descriptive prompts for LLM based on topic words extracted from two PDF documents,
    combining similar topics across both documents and handling parsing errors with retries.
    """

    # Format topic lists into strings for the prompt
    def format_topics(topics, doc_num):
        topics_string = f"Document {doc_num} Topics:\n"
        for i, lst in enumerate(topics):
            topics_string += f"Topic {i+1}: {', '.join(lst)}\n"
        return topics_string

    topics_string_doc1 = format_topics(list_of_topics_doc1, document1)
    topics_string_doc2 = format_topics(list_of_topics_doc2, document2)

    parser = JsonOutputParser(pydantic_object=DataFormat)

    template_string = """
    You are an expert in comparing documents based on common topic keywords. 
    Your task is to analyze the following lists of keywords, each representing a potential topic extracted from two separate documents.

    [IMPORTANT]
    Describe each theme concisely and meaningfully. Combine similar topics across *both* documents into broader themes.
    For each theme, identify 2-4 *meaningful and descriptive subthemes* as short phrases (2-5 words each) that capture the nuances of the combined topic.  
    Do not use single words or words connected by underscores as subthemes. Ensure the subthemes provide specific insights related to the theme.
    [/IMPORTANT]

    Topic from Document 1:
        {topics_string_doc1}
    
    Topic from Document
        {topics_string_doc2}

    Output Format: Provide your response in JSON format.

    If topics are very similar, combine them into a single theme and its associated subthemes with multiple subthemes representing the nuances of the combined topics.
    Ensure the "themes" list in the JSON output reflects these combined themes. MUST Aim for clear and distinct subthemes under each theme, but avoid creating too many subthemes. 
    Consider the context from *both* documents when identifying common themes.

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template_string,
        input_variables=["topics_string_doc1", "topics_string_doc2"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    for attempt in range(max_retries):
        try:
            output = chain.invoke(
                {
                    "topics_string_doc1": topics_string_doc1,
                    "topics_string_doc2": topics_string_doc2,
                }
            )
            return output  # Returns the parsed DataFormat object
        except Exception as e:
            print(f"Error parsing LLM output on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                print(f"Failed to parse output after {max_retries} attempts.")
                return None  # Or raise the error or handle as needed


def get_rag_chain(llm):
    insight_temp = """
    <instruction>
    You are an expert business insights analyst.  
    You will be provided with a theme and a sub-theme within that theme. 
    Your task is to analyze the provided context and extract the most informative insights related to the given sub-theme.  
    Your insights should be specific, quantifiable whenever possible. Prioritize insights that has more importance
    </instruction>

    <context>
    Context: {context}
    </context>

    <theme>
    Theme: {theme}
    </theme>

    <sub_theme>
    Sub-theme: {sub_themes}
    </sub_theme>

    Output:
    """
    prompt = PromptTemplate.from_template(insight_temp)
    rag_chain = prompt | llm
    return rag_chain


def get_theme_insight(llm, retriever_dict, theme_name, subtheme_name, filename):
    # item = insight_data['themes'][index]
    theme = theme_name
    sub_theme = subtheme_name

    query = (
        f"In detail tell me about {theme}, and the sub-themes {', '.join(sub_theme)}"
    )

    if filename not in retriever_dict:
        raise ValueError(f"File '{filename}' not found in the retriever dictionary.")

    retriever = retriever_dict[filename]

    context = format_docs(retriever.invoke(query), file_name=filename)

    rag_chain = get_rag_chain(llm=llm)

    for chunk in rag_chain.stream(
        {"theme": theme, "sub_themes": sub_theme, "context": context}
    ):
        yield chunk.content


######################## comp app


def get_context_with_page_number(context):
    doc_name = extract_document_name(context[0].metadata["file_path"])
    new_context = ""
    new_context = f"""Context from {doc_name} is being given in chunks, each chunk of context consists of the page number from where it has been picked and the page content . Below is the context:"""

    for item in context:
        new_context += f"\n[The below context chunk is from page number: {int(item.metadata['page'])+1}]\n"
        new_context += item.page_content
    return new_context, doc_name


def get_context(retriever_dict, theme, sub_theme):
    query = (
        f"""Find out the relevant documents for the {sub_theme} within the {theme} """
    )

    file_name_1 = list(retriever_dict.keys())[0]  # Assuming you have at least two files
    file_name_2 = list(retriever_dict.keys())[1]

    retriever_1 = retriever_dict[file_name_1]
    retriever_2 = retriever_dict[file_name_2]

    context_1, doc_1_name = get_context_with_page_number(retriever_1.invoke(query))
    context_2, doc_2_name = get_context_with_page_number(retriever_2.invoke(query))

    return (context_1, doc_1_name, context_2, doc_2_name)


from langchain.prompts import PromptTemplate

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class ComparisonOutput1(BaseModel):
    talking_points: list[dict] = Field(
        description="List of talking points, each with doc1 and doc2 perspectives."
    )


def get_detailed_comparison_chain1(llm, theme, sub_theme, retriever_dict):
    context_1, doc_1_name, context_2, doc_2_name = get_context(
        retriever_dict=retriever_dict, theme=theme, sub_theme=sub_theme
    )

    output_parser = PydanticOutputParser(pydantic_object=ComparisonOutput1)

    detailed_insight_prompt = PromptTemplate(
        template="""
        Compare two documents on a specific subtheme and extract comparable talking points.

        **Main Theme:** {theme}
        **Subtheme:** {sub_theme}

        **Document 1: {doc_1_name}**
        Content: {context_1}


        **Document 2: {doc_2_name}**
        Content: {context_2}


        **Instructions:**

        1. **Identify Talking Points:**  Find common themes or aspects discussed in both documents related to the subtheme. These will be your "talking points."

        2. **Extract Perspectives:** For each talking point, summarize the perspective or key information presented in *both* documents.

        3. **Structure Output:** Your output MUST adhere to the following JSON structure:
        
        {format_instructions}

        """,
        input_variables=[
            "theme",
            "sub_theme",
            "context_1",
            "context_2",
            "doc_1_name",
            "doc_2_name",
        ],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )

    comparison_chain = detailed_insight_prompt | llm | output_parser
    output = comparison_chain.invoke(
        {
            "theme": theme,
            "sub_theme": sub_theme,
            "context_1": context_1,
            "context_2": context_2,
            "doc_1_name": doc_1_name,
            "doc_2_name": doc_2_name,
        }
    )

    return (output, doc_1_name, doc_2_name)


def get_detailed_comparison_chain(llm, theme, sub_theme, retriever_dict):

    context_1, doc_1_name, context_2, doc_2_name = get_context(
        retriever_dict=retriever_dict, theme=theme, sub_theme=sub_theme
    )

    output_parser = JsonOutputParser(pydantic_object=ComparisonOutput)
    detailed_insight_prompt = PromptTemplate(
        template="""
        You are tasked with generating detailed insights from the contexts taken from two documents on a specific subtheme and include references.
        Avoid summarizing; instead, provide comprehensive insights.
        context_1 has been extracted from the document named {doc_1_name} and context_2 from {doc_2_name}.
        
        **Main Theme:** {theme}
        **Subtheme:** {sub_theme}
        
        **Document {doc_1_name} has the following contents:** 
        
        {context_1}
        
        **Document {doc_2_name} has the following contents:** 
        
        {context_2}
        
        **Instructions:**
        
        1. **Individual Insights:** For each document, extract all relevant insights related to the subtheme. Go into detail, exploring nuances, implications, and any supporting evidence within the text.
        
        2. **Comparative Insights:** Identify all comparative insights between the documents' treatment of the subtheme. Be specific and quote relevant sections.
        
        3. **Output Structure:** Your output must adhere to the following JSON structure, providing exhaustive detail in each field:
        
        **Output Format Instructions:**
        ```json
        {format_instructions}
        ```
        
        1. Always refer to documents by their provided names.
        
        2. Emphasize key information by using bold text.
        
        3. Provide precise references for all information. The format should be [Page, Paragraph/Section] e.g., [Page 3, Paragraph 2] or [Page 8 - Section 4.1]. References should always be italicized.
        """,
        input_variables=[
            "theme",
            "sub_theme",
            "context_1",
            "context_2",
            "doc_1_name",
            "doc_2_name",
        ],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )

    comparison_chain = detailed_insight_prompt | llm | output_parser
    output = comparison_chain.invoke(
        {
            "theme": theme,
            "sub_theme": sub_theme,
            "context_1": context_1,
            "context_2": context_2,
            "doc_1_name": doc_1_name,
            "doc_2_name": doc_2_name,
        }
    )
    return (output, doc_1_name, doc_2_name)

    # detailed_comparison_prompt = PromptTemplate(
    #     template="""
    #     You are tasked with performing a very detailed comparison between the contexts taken from two documents on a specific subtheme and include references.\
    #     Avoid summarizing; instead, provide comprehensive details.
    #     context_1 has been extracted from the documnent named {doc_1_name}and context_2 from {doc_2_name}.

    #     **Main Theme:** {theme}
    #     **Subtheme:** {sub_theme}

    #     **Document {doc_1_name} has following contents:**

    #     {context_1}

    #     **Document {doc_2_name} has following contents:**

    #     {context_2}

    #     **Instructions:**

    #     1. **Individual Deep Dives:**  For each document, extract all information relevant to the subtheme.  Go into detail, exploring nuances, implications, and any supporting evidence within the text.

    #     2. **Detailed Similarities:** Identify *all* similarities between the documents' treatment of the subtheme. Be specific and quote relevant sections.

    #     3. **Detailed Differences:** Identify *all* differences, no matter how small. Explain the nature of each difference and its potential significance. Again, be specific and quote/reference relevant sections.

    #     4. **Numerical Insight Comparison:** If either document presents numerical data related to the subtheme, compare those data points thoroughly. Highlight trends, discrepancies, and any insights derived from the numerical information. Present this comparison in a clear and structured format (e.g., a markdown table or a detailed textual description).

    #     5. **Output Structure:** Your output must adhere to the following JSON structure, providing exhaustive detail in each field:

    #     **Output Format Instructions:**
    #     ```json
    #     {format_instructions}
    #     ```

    #     1. Always refer to documents by their provided names.

    #     2. Emphasize key information by using bold text.

    #     3. Provide precise references for all information. The format should be [Page, Paragraph/Section] e.g., [Page 3, Paragraph 2] or [Page 8 - Section 4.1]. References should always be italicized.
    #     """,
    #     input_variables=['theme', 'sub_theme', 'context_1', 'context_2','doc_1_name', 'doc_2_name'],
    #     partial_variables={'format_instructions': output_parser.get_format_instructions()}
    # )

    # comparison_chain = detailed_comparison_prompt | llm | output_parser
    # output = comparison_chain.invoke({"theme": theme,"sub_theme": sub_theme, "context_1": context_1, "context_2":context_2, 'doc_1_name': doc_1_name,'doc_2_name': doc_2_name})
    # return (output,doc_1_name,doc_2_name)


def get_comparison_chain_stream():

    output_parser = JsonOutputParser(pydantic_object=Theme)

    comp_prompt = PromptTemplate(
        template="""
    Compare the following two documents and identify the differences between them only on the following themes {subs}. 
    Present the differences in a table with three columns:
    Theme: The main topic or subject of the difference.
    Document 1: The specific content or detail from the first document.
    Document 2: The specific content or detail from the second document.


    Output Format:
    <instruction>
    Only include themes where there are differences between the two documents. Ensure the differences are extremely detailed,directly related to the other document, and include numerical comparisons if available.
    Give the output table in markdown format.
    | Theme       | Document 1 Differences                                                                 | Document 2 Differences                                                                 |
    |-------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
    | [Theme 1]   | [Detailed difference in Document 1 related to Theme 1, in relation to Document 2, including numerical comparisons if available]  | [Detailed difference in Document 2 related to Theme 1, in relation to Document 1, including numerical comparisons if available]  |
    | [Theme 2]   | [Detailed difference in Document 1 related to Theme 2, in relation to Document 2, including numerical comparisons if available]  | [Detailed difference in Document 2 related to Theme 2, in relation to Document 1, including numerical comparisons if available]  |
    | ...         | ...                                                                                    | ...                                                                                    |

    </instruction>

    Please ensure that the sub-themes are clearly listed under each main theme, and the differences are accurately captured for both documents.

    Contexts from the two documents are as follows:
    context_1: {context_1}
    context_2: {context_2}

    """,
        input_variables=["subs", "context_1", "context_2"],
    )

    comp_chain = comp_prompt | st.session_state.llm
    return comp_chain


def get_chat_answer(retriever, query, llm, final_output):
    if retriever is not None and llm is not None:
        test = """
                You are an Expert Financial Analyst.
                Answer the following question based only on the provided context. 

                Think step by step before providing a detailed answer. 

                I will tip you $1000 if the user finds the MOST useful answer and satisfied. 
            <context>

                {context}
            </context>

                Question: {query}"""

        prompt = PromptTemplate.from_template(test)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        final_answer_chain = (
            {"context": retriever[-1] | format_docs, "query": RunnablePassthrough()}
            | prompt
            | llm
        )

        for chunk in final_answer_chain.stream(query):
            final_output += chunk.content
            yield chunk.content

    return final_output


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# @st.cache_data  # Cache create_chatbot_chain
def create_chatbot_chain(retriever, llm):
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know.
    {context}
    Question: {question}"""
    QA_PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )
    return qa_chain


def initialise_embed_llm(em_llm_opn: dict):
    print(f"Model name ---> {em_llm_opn}")
    if em_llm_opn["visible_name"] == "text-embedding-ada-002":
        return get_gpt_embedding()
    else:
        return HuggingFaceEmbeddings(model_name=em_llm_opn, show_progress=True)


def initialise_llm(llm_source: dict, llm_opn: dict):
    print(f"Model name ---> {llm_opn}")
    model = llm_opn
    if llm_source == LLMSource.openai:
        return get_gpt_mini()
    elif llm_source == LLMSource.ollama:
        return ChatOllama(model=model, temperature=0)
    elif llm_source == LLMSource.chatgroq:
        return ChatGroq(
            model=model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
