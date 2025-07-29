import os
import requests
import json
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from helpers.utils import init_env_from_yaml
init_env_from_yaml()

credentials = Credentials(
  url =os.getenv("WXAI_INFER_ENDPOINT"),
  api_key = os.getenv("WXAI_ACCESS_KEY"),
)

client = APIClient(credentials)

params = {
  "decoding_method": "greedy",
  "temperature": 0.5,
  "min_new_tokens": 10,
  "max_new_tokens": 100
}

model_id = "mistralai/mistral-large"
project_id = os.getenv("WXAI_PROJECT_ID")
space_id = None # optional
verify = False

model = ModelInference(
  model_id=model_id,
  api_client=client,
  params=params,
  project_id=project_id,
  space_id=space_id,
  verify=verify,
)


# Read your IBM Cloud API key
API_KEY = os.getenv("WXAI_ACCESS_KEY")  # e.g., from .env or Docker secrets
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Dictionary of agent endpoints
AGENT_ENDPOINTS = {
    "Vector Search Agent": "https://us-south.ml.cloud.ibm.com/ml/v4/deployments/4a7665d0-1fe9-490a-bd13-73efbdecb5bf/ai_service_stream?version=2021-05-01",
    "Data Visualizer Agent": "https://us-south.ml.cloud.ibm.com/ml/v4/deployments/2212c8d4-00af-4c05-8e83-f0b1bb420b78/ai_service_stream?version=2021-05-01",
    "Data Agent": "https://us-south.ml.cloud.ibm.com/ml/v4/deployments/35a030c7-0694-45ad-868a-2181688cd7c3/ai_service_stream?version=2021-05-01",
    "default_agent":""
}

def orchestrator(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "black friday" in prompt_lower:
        return "Data Agent"
    elif "comparaison" in prompt_lower:
        return "Data Visualizer Agent"
    elif "product" or "AirPods" in prompt_lower:
        return "Vector Search Agent"
    else:
        return "default_agent" # Prompt 
    

def get_iam_token(api_key: str) -> str:
    if not api_key:
        raise ValueError("API_KEY not found. Please set WXAI_ACCESS_KEY in your environment.")
    token_data = {
        "apikey": api_key,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
    }
    resp = requests.post(IAM_TOKEN_URL, data=token_data)
    resp.raise_for_status()
    return resp.json()["access_token"]

def stream_watsonx_response(deployment_url: str, prompt: str):
    mltoken = get_iam_token(API_KEY)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {mltoken}"
    }

    payload = {
        "messages": [
            {
                "content": prompt,
                "role": "user"
            }
        ]
    }

    with requests.post(deployment_url, headers=headers, json=payload, stream=True) as r:
        if r.status_code != 200:
            yield f"ERROR {r.status_code}: {r.text}"
            return

        for chunk in r.iter_lines(decode_unicode=True):
            if chunk and chunk.strip().startswith("data: "):
                json_str = chunk[len("data: "):].strip()
                if json_str in ("[DONE]", ""):
                    continue
                try:
                    data_json = json.loads(json_str)
                    delta = data_json["choices"][0]["message"].get("delta", "")
                    yield delta
                except (json.JSONDecodeError, KeyError):
                    continue

def fetch_top_selling_black_friday_data() -> str:
    url = "http://localhost:8080/get_top_selling_black_friday"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            lines = ["Top Selling Products for Black Friday:"]
            for item in data:
                product = item.get("product", "Unknown")
                total_sales = item.get("total_sales")  # Do not supply a default here
                formatted_sales = format_sales(total_sales)
                lines.append(f"{product}: {formatted_sales}")
            return "\n".join(lines)
        else:
            return "Unexpected data format from API."
    except Exception as e:
        return f"Error fetching data: {e}"
    
def format_sales(sales):
    if sales is None:
        return "N/A"
    try:
        sales_float = float(sales)
        return f"${sales_float/1000:.1f}k"
    except Exception:
        return str(sales)

def build_combined_prompt(user_prompt: str, agent: str) -> str:
    if agent == "Data Agent":
        extra_data = fetch_top_selling_black_friday_data()
        instructions = (
            "You are a helpful Data Agent specializing in retail sales analysis."
            "Answer ONLY using the data provided. Do not add any additional products, details, or assumptions beyond the data."
            "If the data does not provide an answer, simply state so.\n\n"
        )
        return f"{instructions}\n\nBlack Friday Sales Data:\n{extra_data}\n{user_prompt}"
    else:
        return user_prompt

def data_agent_retrieve(prompt):
    response = model.generate_text_stream(prompt, params)
    return response


def fetch_market_analysis_data():
    url = "http://localhost:8080/get_market_analysis"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("analysis_results", [])
    except Exception as e:
        e.error(f"Error fetching market analysis data: {e}")
        return []
