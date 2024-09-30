import pandas as pd

from chat_llm import ChatLlamaAPI
from utils.chain import *
from utils.load_data import wrap_input_for_demo

statement = "the wildcats kept the opposing team scoreless in four games"
table_caption = "1947 kentucky wildcats football team"
table_text = [
    ['game', 'date', 'opponent', 'result', 'wildcats points', 'opponents', 'record'],
    ['1', 'sept 20', 'ole miss', 'loss', '7', '14', '0 - 1'],
    ['2', 'sept 27', 'cincinnati', 'win', '20', '0', '1 - 1'],
    ['3', 'oct 4', 'xavier', 'win', '20', '7', '2 - 1'],
    ['4', 'oct 11', '9 georgia', 'win', '26', '0', '3 - 1 , 20'],
    ['5', 'oct 18', '10 vanderbilt', 'win', '14', '0', '4 - 1 , 14'],
    ['6', 'oct 25', 'michigan state', 'win', '7', '6', '5 - 1 , 13'],
    ['7', 'nov 1', '18 alabama', 'loss', '0', '13', '5 - 2'],
    ['8', 'nov 8', 'west virginia', 'win', '15', '6', '6 - 2'],
    ['9', 'nov 15', 'evansville', 'win', '36', '0', '7 - 2'],
    ['10', 'nov 22', 'tennessee', 'loss', '6', '13', '7 - 3']
]
answer = "True"


llama_api_token = "LA-1e8d28e25128414ea99e2faa7cd8e2e9e4fc2f41804c4b5ba189fd9a6e8f142c"


LLM = ChatLlamaAPI(
    model_name="llama3.1-70b",
    key=llama_api_token,
)


demo_sample = wrap_input_for_demo(
    statement=statement, table_caption=table_caption, table_text=table_text
)

proc_sample, dynamic_chain_log = dynamic_chain_exec_one_sample(
    sample=demo_sample, llm=LLM
)
print(f'Statements: {proc_sample["statement"]}\n')
print(f'Table: {proc_sample["table_caption"]}')

output_sample = simple_query(
    sample=proc_sample,
    table_info=get_table_info(proc_sample),
    llm=LLM,
    use_demo=True,
    llm_options=LLM.get_model_options(
        temperature=0.0, per_example_max_decode_steps=200, per_example_top_p=1.0
    ),
)
cotable_log = get_table_log(output_sample)


print(f"{pd.DataFrame(table_text[1:], columns=table_text[0])}\n")
for table_info in cotable_log:
    if table_info["act_chain"]:
        table_text = table_info["table_text"]
        table_action = table_info["act_chain"][-1]
        if "skip" in table_action:
            continue
        if "query" in table_action:
            result = table_info["cotable_result"]
            if result == "YES":
                print(f"-> {table_action}\nThe statement is True\n")
            else:
                print(f"-> {table_action}\nThe statement is False\n")
        else:
            print(f"-> {table_action}\n{pd.DataFrame(table_text[1:], columns=table_text[0])}")
            if 'group_sub_table' in table_info:
                group_column, group_info = table_info["group_sub_table"]
                group_headers = ["Group ID", group_column, "Count"]
                group_rows = []
                for i, (v, count) in enumerate(group_info):
                    if v.strip() == "":
                        v = "[Empty Cell]"
                    group_rows.append([f"Group {i+1}", v, str(count)])
                print(f"{pd.DataFrame(group_rows, columns=group_headers)}")
            print()

print(f"Groundtruth: The statement is {answer}")
