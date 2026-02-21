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

response = client.chat.completions.create(
  model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": "Who were the founders of Microsoft?"}
    ]
)

#print(response)
print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)