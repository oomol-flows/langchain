import re

from typing import Any, Dict, List
from oocana import Context
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import BaseCallbackHandler

class BuiltinCallbackHandler(BaseCallbackHandler):
  def __init__(self, context: Context) -> None:
    super().__init__()
    self._context: Context = context

  def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **_: Any) -> Any:
    self._print_splitter()
    self._print_content("Input to Chain", inputs)
    print("\n")

  def on_chain_end(self, outputs: Dict[str, Any], **_: Any) -> Any:
    self._print_splitter()
    self._print_content("Output From Chain", outputs)
    print("\n")

  def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **_: Any) -> Any:
    self._print_splitter()
    print("Chat Model Start:")
    for sub_messages in messages:
      for message in sub_messages:
        text = self._value_to_str(message)
        if self._is_multi_lines(text):
          print("  - \"\"\"")
          print(text)
          print("\"\"\"\n")
        else:
          print(f"  - {text}")

    print("\n")

  def on_llm_end(self, response: LLMResult, **_: Any) -> Any:
    self._print_splitter()
    output = response.llm_output
    if output is not None:
      self._print_content("LLM Output", output)
      print("\n")
      token_usage: dict = output.get("token_usage", None)
      if token_usage is not None:
        completion_tokens = token_usage.get("completion_tokens", 0)
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        data_lines: list[str] = [
          "### Tokens Costs",
          f"- **completion**: {completion_tokens}",
          f"- **prompt**: {prompt_tokens}",
          f"- **total**: {total_tokens}",
        ]
        self._context.preview({
          "type": "markdown",
          "data": "\n\n".join(data_lines),
        })
    else:
      print("LLM Output with Nothing")

  def _print_splitter(self):
    print("--------------------------------")

  def _print_content(self, field_name: str, content: Dict[str, Any] | List[Any] | str):
    if isinstance(content, dict):
      multi_lines: dict[str, str] = {}
      single_lines: dict[str, str] = {}

      for key, value in content.items():
        text = self._value_to_str(value)
        if self._is_multi_lines(text):
          multi_lines[key] = text
        else:
          single_lines[key] = text

      print(f"{field_name}:")
      keys = list(single_lines.keys())
      keys.sort()

      for key in keys:
        print(f"  {key}: {single_lines[key]}")

      keys = list(multi_lines.keys())
      keys.sort()

      for key in keys:
        print(f"  {key}: \"\"\"")
        print(multi_lines[key])
        print("\"\"\"\n")

    elif isinstance(content, list):
      print(f"{field_name}:")
      for index, value in enumerate(content):
        text = self._value_to_str(value)
        if self._is_multi_lines(text):
          print(f"  - [{index}]: \"\"\"")
          print(text)
          print("\"\"\"\n")
        else:
          print(f"  - [{index}]: {text}")

    elif self._is_multi_lines(content):
      print(f"{field_name}: \"\"\"")
      print(content)
      print("\"\"\"\n")
    else:
      print(f"{field_name}: {content}")

  def _value_to_str(self, value: Any) -> str:
    text: str = ""
    if isinstance(value, str):
      text = self._strip(value)

    elif isinstance(value, BaseMessage):
      content = value.content
      if isinstance(content, str):
        text = content
      else:
        text = "".join([str(it) for it in content])

      text = self._strip(text)
      role = ""

      if isinstance(value, SystemMessage):
        role = "System"
      elif isinstance(value, HumanMessage):
        role = "Human"
      elif isinstance(value, AIMessage):
        role = "AI"
      else:
        role = "Unknown"
      text = f"{role}:\n{text}"

    else:
      text = self._strip(str(value))

    return text

  def _is_multi_lines(self, text: str) -> bool:
    return re.search(r"\n", text) is not None

  def _strip(self, text: str) -> str:
    return re.sub(r"^[\s\n]+|[\s\n]+$", "", text)
