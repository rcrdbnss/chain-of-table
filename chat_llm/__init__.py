class ChatLLM:
    def __init__(self, model_name, key):
      self.model_name = model_name
      self.key = key

    def messages(self, prompt):
      return [
          {
              "role": "system",
              "content": "I will give you some examples, you need to follow the examples and complete the text, and no other content.",
          },
          {"role": "user", "content": prompt},
      ]

    def get_model_options(
        self,
        temperature=0,
        per_example_max_decode_steps=150,
        per_example_top_p=1,
        n_sample=1,
    ) -> dict:
        pass

    def generate_plus_with_score(self, prompt, options=None, end_str=None) -> list[tuple[str, float]]:
        pass

    def generate(self, prompt, options=None, end_str=None) -> str:
        pass


from chat_llm.llama_api import ChatLlamaAPI
