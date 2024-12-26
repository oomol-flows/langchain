from langchain_core.runnables import RunnablePassthrough

def main(params: dict):
  retriever = params["retriever"]
  output: dict = {
    "context": retriever | format_docs,
    "input": RunnablePassthrough(),
  }
  return { "output": output }

def format_docs(docs):
  return "\n\n".join(docs)
