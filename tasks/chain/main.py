from typing import Any
from oocana import Context
from pydantic import BaseModel

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables import RunnableSequence
from langchain.agents import AgentExecutor
from .callback import BuiltinCallbackHandler

def main(inputs: dict, context: Context):
  chain: RunnableSerializable = inputs["chain"]
  if isinstance(chain, RunnableSequence):
    return _invoke_runnable(chain, inputs, context)
  elif isinstance(chain, AgentExecutor):
    return _invoke_agent_executor(chain, inputs, context)
  else:
    raise ValueError(f"Unsupported chain type: {type(chain)}")

def _invoke_runnable(chain: RunnableSequence, inputs: dict, context: Context):
  params: Any | None  = inputs["params"]
  response_structured = chain.invoke(
    input=params, 
    config={
    "callbacks": [BuiltinCallbackHandler(context)]
    },
  )
  response = response_structured

  if isinstance(response, BaseModel):
    response = response.model_dump()

  return {
    "response": str(response),
    "response_structured": response_structured,
  }

def _invoke_agent_executor(executor: AgentExecutor, inputs: dict, context: Context):
  params = inputs["params"]
  if not isinstance(params, str):
    raise ValueError(f"expect string params (got {type(params)}) when chain is a agent executor")
  params = {"input": params}
  response = executor.invoke(params, {
    "callbacks": [BuiltinCallbackHandler(context)]
  })
  return {
    "response": response["output"],
  }