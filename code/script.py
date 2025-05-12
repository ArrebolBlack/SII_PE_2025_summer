import json
import random
import jsonlines

# 文件路径
val_path = "E:\PE_Exam\\val.jsonl"

# 输出结构
final_dataset = []

def load_jsonl(filepath):
    with jsonlines.open(filepath) as reader:
        return list(reader)

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)



print()



