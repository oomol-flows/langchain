from langchain_core.prompts import PromptTemplate

def main(params: dict):
  template: PromptTemplate = params["template"]
  input: dict[str, str] = params["input"]
  output = template.invoke(input)
  return { "prompt": output }