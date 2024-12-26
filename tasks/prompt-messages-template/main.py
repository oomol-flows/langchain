from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.prompts import PromptTemplate

def main(params: dict):
  messages: list[dict] = params["messages"]
  input: RunnableSerializable | None = params["input"]
  output: RunnableSerializable = ChatPromptTemplate([
    (m["role"], m["template"]) for m in messages
  ])
  if input is not None:
    output = input | output

  return { "output": output }
