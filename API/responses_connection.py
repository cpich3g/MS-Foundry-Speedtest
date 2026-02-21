from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

import os
from dotenv import load_dotenv
load_dotenv()

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = OpenAI(  
  base_url = os.getenv("AZURE_FOUNDRY_ENDPOINT"),  
  api_key=token_provider,
)

response = client.responses.create(
    model="gpt-4.1-nano",
    input= "This is a test" 
)

print(response.model_dump_json(indent=2))