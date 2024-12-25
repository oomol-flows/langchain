from langchain_core.runnables import RunnableSerializable
from langchain_core.prompts import PromptTemplate

def main(params: dict):
  input: RunnableSerializable | None = params["input"]
  template: str = params["template"]
  output: RunnableSerializable = PromptTemplate.from_template(template)
  if input is not None:
    output = input | output
  return { "output": None }
