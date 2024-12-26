from oocana import Context
from pydantic import BaseModel

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables import RunnableSequence
from langchain_core.prompt_values import PromptValue
from langchain.agents import AgentExecutor
from .callback import BuiltinCallbackHandler

def main(params: dict, context: Context):
  output: RunnableSerializable = params["output"]
  if isinstance(output, RunnableSequence):
    return _invoke_runnable(output, params, context)
  elif isinstance(output, AgentExecutor):
    return _invoke_agent_executor(output, params, context)
  else:
    raise ValueError(f"Unsupported chain type: {type(output)}")

def _invoke_runnable(output: RunnableSequence, params: dict, context: Context):
  prompt: PromptValue | None  = params["prompt"]
  response_structured = output.invoke(
    input=prompt, 
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

def _invoke_agent_executor(executor: AgentExecutor, params: dict, context: Context):
  prompt: str | dict = params["prompt"]
  if not isinstance(prompt, str):
    raise ValueError(f"expect string params (got {type(prompt)}) when chain is a agent executor")
  prompt = {"input": prompt}
  response = executor.invoke(prompt, {
    "callbacks": [BuiltinCallbackHandler(context)]
  })
  return {
    "response": response["output"],
  }