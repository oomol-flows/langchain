from numpy import ndarray
from typing import cast, Literal

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

from oocana import Context

_Device = Literal["cpu", "cuda"]

def main(params: dict, context: Context):
  id: str = params["id"]
  model_dir: str | None = params["model_dir"]
  query_template: str | None = params["query_template"]
  device: _Device = params["device"]

  if query_template is None:
    query_template = "{input}"

  embeddings = _BuiltinEmbeddings(
    model_id=id, 
    query_template=query_template,
    model_dir=model_dir,
    device = device,
  )
  return { "output": embeddings }

class _BuiltinEmbeddings(Embeddings):
  def __init__(
      self, 
      model_id: str, 
      query_template: str,
      model_dir: str | None,
      device: _Device,
    ):
    self._query_template: str = query_template
    self._model = SentenceTransformer(
      model_name_or_path=model_id,
      device=device,
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