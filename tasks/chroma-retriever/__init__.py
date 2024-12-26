import os
import shutil

from oocana import Context
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

def main(params: dict, context: Context):
  documents: list[Document] = params["documents"]
  database_path: str | None = params["database_path"]
  embeddings: Embeddings | None = params["embeddings"]
  search_k: int = params["search_k"]
  reset_database_at_beginning: bool = params["reset_database_at_beginning"]
  vector_store: Chroma

  if database_path is None:
    database_path = os.path.join(
      context.session_dir, "chroma_db",
    )
  if documents is None:
    vector_store = Chroma(
      persist_directory=database_path,
      embedding_function=embeddings,
    )
  else:
    # https://python.langchain.com/v0.2/docs/tutorials/rag/
    if reset_database_at_beginning:
      shutil.rmtree(database_path, ignore_errors=True)
    vector_store = Chroma.from_documents(
      documents=documents,
      persist_directory=database_path,
      embedding=embeddings,
    )
  output = vector_store.as_retriever(
    search_kwargs={"k": search_k},
  )
  return { "output": output }