# FastAPI Endpoint

from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

class APIPaths:
    # api endpoints

    # get api

    # post api
    # api_url = "http://192.168.74.165:8000/"
    api_url=os.environ["backend_url"]
    # api_url="http://localhost:8000/"
    print(">>>>>>>>>>>>>>>>>>>", api_url)
    init = api_url + "init/"
    download_model = api_url + "download_model/"
    save_documents = api_url + "save_documents/"
    process_document = api_url + "process_documents/"
    get_theme_insights = api_url + "get_theme_insights/"
    get_comparison = api_url + "get_comparison/"
    get_chat_response = api_url + "get_chat_response/"
