from langchain_core.documents import Document

def main(params: dict):
  url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
  texts: list[str] = params["texts"]
  documents: list[Document] = [
    Document(
      page_content=text,
      metadata={ "source": url },
    )
    for text in texts
  ]
  return { "documents": documents }
