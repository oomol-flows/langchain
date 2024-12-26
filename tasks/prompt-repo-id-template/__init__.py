from langchain import hub
from langchain_core.runnables import RunnableSerializable
from langchain_core.prompts import PromptTemplate

def main(params: dict):
  owner_repo_commit: str = params["id"]
  input: RunnableSerializable | None = params["params"]
  output = hub.pull(owner_repo_commit)
  if input is not None:
    output = input | output

  return { "template": output }