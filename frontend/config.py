import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# folder paths
INDEX = os.path.join(ROOT_DIR, "index")
VECTOR_STORE = os.path.join(INDEX, "vector_store")
JSON_STORE = os.path.join(INDEX, "json_store")
UPLOAD_DIRECTORY = os.path.join(INDEX, "uploads")
