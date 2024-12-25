from typing import Literal
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI

def main(params: dict):
  return { "output": build_model(params) }

def build_model(params: dict) -> ChatOpenAI | ChatAnthropic | ChatVertexAI:
  interface: Literal["openai", "claude", "gemni"] = params["interface"]
  api_key: SecretStr = params["api_key"]
  base_url: str | None = params["base_url"]
  model_name: str = params["model"]
  temperature: float = params["temperature"]
  timeout: float = params["timeout"]

  if interface == "openai":
    if api_key is None:
      raise ValueError("API key is required for OpenAI interface")
    return ChatOpenAI(
      api_key=api_key,
      base_url=base_url,
      model=model_name,
      temperature=temperature,
      timeout=timeout,
    )
  elif interface == "claude":
    if api_key is None:
      raise ValueError("API key is required for Claude interface")
    return ChatAnthropic(
      api_key=api_key,
      model_name=model_name,
      base_url=base_url,
      timeout=timeout,
      stop=None,
      temperature=temperature,
    )
  elif interface == "gemni":
    return ChatVertexAI(
      model=model_name,
      base_url=base_url,
      timeout=timeout,
      temperature=temperature,
    )
  else:
    raise ValueError(f"Unsupported interface: {interface}")