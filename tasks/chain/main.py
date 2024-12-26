from oocana import Context
from pydantic import BaseModel

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables import RunnableSequence
from langchain_core.prompt_values import PromptValue
from langchain.agents import AgentExecutor
from .callback import BuiltinCallbackHandler

def main(inputs: dict, context: Context):
  model: RunnableSerializable = inputs["model"]
  if isinstance(model, RunnableSequence):
    return _invoke_runnable(model, inputs, context)
  elif isinstance(model, AgentExecutor):
    return _invoke_agent_executor(model, inputs, context)
  else:
    raise ValueError(f"Unsupported chain type: {type(model)}")

def _invoke_runnable(model: RunnableSequence, inputs: dict, context: Context):
  input: PromptValue | None  = inputs["input"]
  response_structured = model.invoke(
    input=input, 
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