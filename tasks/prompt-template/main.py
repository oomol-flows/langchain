from langchain_core.prompts import PromptTemplate

def main(params: dict):
  return { "output": PromptTemplate.from_template(params["template"]) }
