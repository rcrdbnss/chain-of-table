# Copyright 2024 The Chain-of-Table authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

from chat_llm import ChatLlamaAPI
from utils.load_data import load_tabfact_dataset
from utils.llm import ChatGPT
from utils.helper import *
from utils.evaluate import *
from utils.chain import *
from operations import *


def main(
    dataset_path: str = "data/tabfact/data_dev.jsonl",
    raw2clean_path: str ="data/tabfact/raw2clean_dev.jsonl",
    model_name: str = "llama3.1-70b",
    result_dir: str = "results/tabfact",
    openai_api_key: str = "LA-1e8d28e25128414ea99e2faa7cd8e2e9e4fc2f41804c4b5ba189fd9a6e8f142c",
    first_n=-1,
    n_proc=1,
    chunk_size=1,
):
    dataset = load_tabfact_dataset(dataset_path, raw2clean_path, first_n=first_n)
    LLM = ChatLlamaAPI(
        model_name=model_name,
        key=os.environ["OPENAI_API_KEY"] if openai_api_key is None else openai_api_key,
    )
    os.makedirs(result_dir, exist_ok=True)

    proc_samples, dynamic_chain_log_list = dynamic_chain_exec_with_cache_for_loop(
        dataset,
        llm=LLM,
        llm_options=LLM.get_model_options(
            temperature=0.0, per_example_max_decode_steps=200, per_example_top_p=1.0
        ),
        strategy="top",
        cache_dir=os.path.join(result_dir, "cache"),
        # n_proc=n_proc,
        # chunk_size=chunk_size,
    )
    fixed_chain = [
        (
            "simpleQuery_fewshot",
            simple_query,
            dict(use_demo=True),
            dict(
                temperature=0, per_example_max_decode_steps=200, per_example_top_p=1.0
            ),
        ),
    ]
    final_result, _ = fixed_chain_exec(LLM, proc_samples, fixed_chain)
    acc = tabfact_match_func_for_samples(final_result)
    print("Accuracy:", acc)

    print(
        f'Accuracy: {acc}',
        file=open(os.path.join(result_dir, "result.txt"), "w")
    )
    pickle.dump(
        final_result, open(os.path.join(result_dir, "final_result.pkl"), "wb")
    )
    pickle.dump(
        dynamic_chain_log_list, 
        open(os.path.join(result_dir, "dynamic_chain_log_list.pkl"), "wb")
    )


if __name__ == "__main__":
    main()
