from langchain_core.runnables import RunnableSerializable
from langchain_core.prompts import PromptTemplate

def main(params: dict):
  input: RunnableSerializable | None = params["params"]
  output: RunnableSerializable = PromptTemplate.from_template(
    template=params["template"]
  )
  if input is not None:
    output = input | output

  return { "template": output }
