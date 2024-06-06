from langchain.prompts.prompt import PromptTemplate

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2#instruction-format
mistral = """<s> [INST] <<SYS>> {system} <</SYS>>
{user}
[/INST] """

from toolz import partial
from langchain.prompts.prompt import PromptTemplate

def generate_prompt(preset: str, system: str, user: str):
    template = PromptTemplate.from_template(preset)
    return template.format(system=system, user=user)

mistral_prompt_builder = partial(generate_prompt, mistral)

def rag_prompt(context:str, query:str):
    prompt_template = PromptTemplate.from_template("""
The context is delimited in backticks below. answer the query.  if you don't know the answer, say you don't know.

```
{context}
```

{query}
""".strip())
    return prompt_template.format(context=context, query=query)

class Rag:
    def __init__(self, config):
        self.llm = config.llm
        self.collection = config.collection

    def query(self, query, n_results=3):
        docs = self.collection.query(query, n_results)
        if docs:
            return self._rag(docs, query)

    def _rag(self, context, query, query_decorator=mistral_prompt_builder):
        user_input = rag_prompt(context, query)
        prompt = query_decorator("You are a helpful assistant.", user_input)
        return self.llm.invoke(prompt)