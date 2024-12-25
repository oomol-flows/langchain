from typing import Any
from langchain_core.prompts import PromptTemplate

def main(params: dict):
  input_params: dict[str, Any] = params["params"]
  template = PromptTemplate.from_template(params["template"])
  prompt = template.invoke(**input_params)
  return { "prompt": prompt }
