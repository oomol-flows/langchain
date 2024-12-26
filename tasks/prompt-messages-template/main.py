from langchain_core.prompts import ChatPromptTemplate

def main(params: dict):
  messages: list[dict] = params["messages"]
  template = ChatPromptTemplate([
    (m["role"], m["template"]) for m in messages
  ])
  return { "output": template }
