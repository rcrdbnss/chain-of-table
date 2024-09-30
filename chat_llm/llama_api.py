import json
import time

import numpy as np
from openai import OpenAI

from chat_llm import ChatLLM


class ChatLlamaAPI(ChatLLM):
    def __init__(self, model_name, key):
        super().__init__(model_name, key)
        self.client = OpenAI(
            api_key=key,
            base_url="https://api.llama-api.com"
        )

    def get_model_options(
        self,
        temperature=0,
        per_example_max_decode_steps=150,
        per_example_top_p=1,
        n_sample=1,
    ):
        return dict(
            temperature=temperature,
            n=n_sample,
            top_p=per_example_top_p,
            max_tokens=per_example_max_decode_steps,
        )

    def generate_plus_with_score(self, prompt, options=None, end_str=None):
        if options is None:
            options = self.get_model_options()

        messages = self.messages(prompt)

        responses = None
        retry_num = 0
        retry_limit = 2
        error = None
        while responses is None:
            try:
                responses = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stop=end_str,
                    **options
                )
                error = None
            except Exception as e:
                error = str(e)
                print(error, flush=True)
                if retry_num > retry_limit:
                    error = "too many retry times"
                    responses = {
                        "message": {"content": "PLACEHOLDER"}
                    }
                else:
                    time.sleep(60)
                retry_num += 1
        if error:
            raise Exception(error)
        results = []
        responses = json.loads(responses.json())
        for i, res in enumerate(responses["choices"]):
            text = res["message"]["content"]
            fake_conf = (len(responses["choices"]) - i) / len(
                responses["choices"]
            )
            results.append((text, np.log(fake_conf)))

        return results

    def generate(self, prompt, options=None, end_str=None):
        if options is None:
            options = self.get_model_options()
        options["n"] = 1
        result = self.generate_plus_with_score(prompt, options, end_str)[0][0]
        return result
