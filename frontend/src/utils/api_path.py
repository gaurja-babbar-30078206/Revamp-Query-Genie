# FastAPI Endpoint


class APIPaths:
    # api endpoints

    # get api

    # post api
    # api_url = "http://192.168.74.165:8000/"
    api_url="http://host.docker.internal:8000/"
    init = api_url + "init/"
    download_model = api_url + "download_model/"
    save_documents = api_url + "save_documents/"
    process_document = api_url + "process_documents/"
    get_theme_insights = api_url + "get_theme_insights/"
    get_comparison = api_url + "get_comparison/"
    get_chat_response = api_url + "get_chat_response/"
