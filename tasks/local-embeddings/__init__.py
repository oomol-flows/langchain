from numpy import ndarray
from typing import cast, Literal
from oocana import Context
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

def main(params: dict, context: Context):
  id: str = params["id"]
  model_dir: str | None = params["model_dir"]
  query_template: str | None = params["query_template"]

  if query_template is None:
    query_template = "{input}"

  embeddings = _BuiltinEmbeddings(
    model_id=id, 
    query_template=query_template,
    model_dir=model_dir,
  )
  return { "embeddings": embeddings }

class _BuiltinEmbeddings(Embeddings):
  def __init__(
      self, 
      model_id: str, 
      query_template: str,
      model_dir: str | None,
    ):
    self._query_template: str = query_template
    self._model = SentenceTransformer(
      model_name_or_path=model_id,
      cache_folder=model_dir,
    )

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    embed_texts: list[list[float]] = []
    for text in texts:
      embed_texts.append(self.embed_query(text))
    return embed_texts

  def embed_query(self, text: str | dict[str, str]) -> list[float]:
    if isinstance(text, dict):
      text = self._query_template.format(**text)
    embeddings = self._model.encode(text)
    return cast(ndarray, embeddings).tolist()