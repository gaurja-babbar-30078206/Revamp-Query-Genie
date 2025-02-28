import json

class Models:
    """For Embedding models and LLM info capture"""
    visible_name = ""
    info = {}

    def __init__(self, visible_name=None, info=None) -> None:
        self.visible_name = visible_name
        self.info = info

class Response():
    """Universal output format accross all APIs"""
    status: str
    data: Models
    message: str

    def __init__(self, status, data, message):
        self.status = status
        self.data = data
        self.message = message


    def serialize(self):
        """Serializes the Response object to a JSON string."""

        return json.dumps({
            "status": self.status,
            "data": {
                "visible_name": self.data.visible_name if self.data else None,  # Handle potential None data
                "info": self.data.info if self.data else None
            },
            "message": self.message
        }, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o) )  # Handle non-serializable objects



    @staticmethod
    def deserialize(json_string):
        """Deserializes a JSON string to a Response object."""
        try:
            data_dict = json.loads(json_string)

            # Safely extract data, handling potential missing keys:
            status = data_dict.get("status")
            message = data_dict.get("message")

            data_info = data_dict.get("data")
            data_obj = None

            if data_info:  # Check if 'data' exists and isn't None
               data_obj = Models(data_info.get("visible_name"), data_info.get("info"))
            
            return Response(status, data_obj, message)


        except json.JSONDecodeError as e:
            # Handle JSON decoding errors (e.g., invalid JSON string)
            print(f"Error deserializing JSON: {e}")
            return None # Or raise an exception, depending on your needs