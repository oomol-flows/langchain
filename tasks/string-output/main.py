from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser


def main(params: dict):
  input: RunnableSerializable | None = params["input"]
  output: RunnableSerializable = StrOutputParser()
  if input is not None:
    output = input | output
  return {
    "output": output,
  }
