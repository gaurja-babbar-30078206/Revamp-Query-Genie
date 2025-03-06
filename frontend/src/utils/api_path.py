from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

class APIPaths:
    api_url = os.environ.get("backend_url", "").rstrip("/") + "/"  # ✅ Ensure trailing slash
    print(">>>>>>>>>>>>>>>>>>> API URL:", api_url)

    # Define API paths
    init = api_url + "init/"
    download_model = api_url + "download_model/"
    save_documents = api_url + "save_documents/"
    process_document = api_url + "process_documents/"
    get_theme_insights = api_url + "get_theme_insights/"
    get_comparison = api_url + "get_comparison/"
    get_chat_response = api_url + "get_chat_response/"
