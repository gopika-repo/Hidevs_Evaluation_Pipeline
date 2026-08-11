import os
from pydantic import BaseModel
from google import genai
from google.genai.types import GenerateContentConfig

class MyResponse(BaseModel):
    hello: str

print("Config with BaseModel:", GenerateContentConfig(response_mime_type="application/json", response_schema=MyResponse))
print("Config with dict:", GenerateContentConfig(response_mime_type="application/json", response_schema=MyResponse.model_json_schema()))
