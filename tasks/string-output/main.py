from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser


def main(params: dict):
  model: RunnableSerializable = params["model"]
  output: RunnableSerializable = StrOutputParser()
  return {
    "output": model | output,
  }
