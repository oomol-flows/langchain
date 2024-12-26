from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

def main(params: dict):
  retriever: VectorStoreRetriever = params["retriever"]
  output: dict = {
    "context": retriever | format,
    "question": RunnablePassthrough(),
  }
  return { "params": output }

def format(docs: list[Document]) -> str:
  return "\n\n".join([d.page_content for d in docs])
