import json
import asyncio
import math
import logging
from tqdm import tqdm
from openai import AsyncOpenAI
import re
import csv

from Prompt_Reranking_Eval import (
    calculate_ndcg_for_sample,
    parse_output,
    query_deepseek_async,
)


import os
from datetime import datetime
log_dir = "logs_Strategy"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"evaluation_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        # logging.StreamHandler()  # 可选：保留控制台输出
    ]
)


# # 多策略提示生成器
# def construct_prompt_variant(d, strategy):
#     system_roles = {
#         "expert": "你是一名电影推荐专家，目标是根据用户历史预测他们最可能观看的电影。",
#         "analyst": "你是一名数据分析师，请基于用户历史行为重排候选电影。"
#     }
#
#     styles = {
#         "instruction": "请根据用户的兴趣偏好，对下列候选电影重排，最可能观看的放最前，仅返回电影ID列表：",
#         "question": "用户接下来最有可能观看哪些电影？请从以下候选中排序，只返回ID列表："
#     }
#
#     candidates_fmt = {
#         "id_first": lambda c: "\n".join([f"{i[0]}: {i[1]}" for i in c]),
#         "name_first": lambda c: "\n".join([f"{i[1]} (ID:{i[0]})" for i in c])
#     }
#
#     history_fmt = {
#         "timestamped": lambda h: "\n".join([f"[{i+1}] {x[1]}" for i, x in enumerate(h)]),
#         "plain": lambda h: "\n".join([f"- {x[1]}" for x in h])
#     }
#
#     system_prompt = system_roles[strategy['role']]
#     user_instruction = styles[strategy['style']]
#     history = history_fmt[strategy['history']](d['item_list'][-10:])
#     candidates = candidates_fmt[strategy['c_format']](d['candidates'])
#
#     if strategy.get("extra") == "comma":
#         user_instruction += "\\n\\n输出格式：用英文逗号分隔一行列出电影ID，不要有空格或额外文字。"
#     elif strategy.get("extra") == "json":
#         user_instruction += "\\n\\n输出格式：仅返回 JSON 数组，如 [2492,684,…]。"
#     elif strategy.get("extra") == "cot":
#         user_instruction += "\\n\\n先在心里思考，不要输出思考过程，仅给出最终 ID 列表。"
#     elif strategy.get("extra") == "example":
#         example = """示例:
#     历史：- A
#     候选：1: X, 2: Y
#     输出：2, 1
#
#     现在请根据以下信息排序："""
#     prompt = example + prompt
#
#     prompt = f"用户最近观看的电影：\n{history}\n\n{user_instruction}\n{candidates}"
#
#     return [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": prompt}
#     ]


def construct_prompt_variant(d, strategy):
    """根据策略组合生成 Prompt 消息列表（修复换行与重复返回Bug）"""
    system_roles = {
        "expert": "你是一名电影推荐专家，目标是根据用户历史预测他们最可能观看的电影。",
        "analyst": "你是一名数据分析师，请基于用户历史行为重排候选电影。"
    }

    styles = {
        "instruction": "请根据用户的兴趣偏好，对下列候选电影重排，最可能观看的放最前，仅返回电影ID列表：",
        "question": "用户接下来最有可能观看哪些电影？请从以下候选中排序，只返回ID列表："
    }

    candidates_fmt = {
        "id_first": lambda c: "\n".join([f"{i[0]}: {i[1]}" for i in c]),
        "name_first": lambda c: "\n".join([f"{i[1]} (ID:{i[0]})" for i in c])
    }

    history_fmt = {
        "timestamped": lambda h: "\n".join([f"[{idx+1}] {m[1]}" for idx, m in enumerate(h)]),
        "plain": lambda h: "\n".join([f"- {m[1]}" for m in h])
    }

    # ---- 组装各部分 ----
    system_prompt = system_roles[strategy['role']]
    user_instruction = styles[strategy['style']]

    extra = strategy.get("extra", "")
    if extra == "comma":
        user_instruction += "\n\n输出格式：用英文逗号分隔一行列出电影ID，不要有空格或额外文字。"
    elif extra == "json":
        user_instruction += "\n\n输出格式：仅返回 JSON 数组，如 [2492,684,…]。"
    elif extra == "cot":
        user_instruction += "\n\n先在心里思考，不要输出思考过程，仅给出最终 ID 列表。"

    history_text = history_fmt[strategy['history']](d['item_list'][-10:])
    candidates_text = candidates_fmt[strategy['c_format']](d['candidates'])

    prompt_body = (
        f"用户最近观看的电影：\n{history_text}\n\n"
        f"{user_instruction}\n{candidates_text}"
    )

    if extra == "example":
        example_block = (
            "示例:\n"
            "历史：- A\n"
            "候选：1: X, 2: Y\n"
            "输出：2,1\n\n"
            "现在请根据以下信息排序：\n"
        )
        prompt_body = example_block + prompt_body

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_body}
    ]


# 异步评测函数
async def evaluate_strategy(val_data, api_key, strategy, max_concurrency=10):
    logging.info(f"\n===== 评估策略: {strategy['name']} =====")
    semaphore = asyncio.Semaphore(max_concurrency)
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    async def worker(d):
        async with semaphore:
            prompt = construct_prompt_variant(d, strategy)
            try:
                output_text = await query_deepseek_async(client, prompt)
                predicted_list = parse_output(output_text)
                return calculate_ndcg_for_sample(predicted_list, d["target_item"][0])
            except Exception as e:
                logging.warning(f"出错样本: {e}")
                return 0.0

    tasks = [worker(d) for d in val_data]
    scores = [await t for t in tqdm(asyncio.as_completed(tasks), total=len(tasks))]
    avg_ndcg = sum(scores) / len(scores)
    logging.info(f"策略 {strategy['name']} 平均 NDCG@10: {avg_ndcg:.4f}\n")
    return strategy['name'], avg_ndcg

# 主运行函数
async def run_all_strategies(val_path, api_key):
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line.strip()) for line in f]

    strategies = []
    for role in ['expert', 'analyst']:
        for style in ['instruction', 'question']:
            for c_format in ['id_first', 'name_first']:
                for history in ['plain', 'timestamped']:
                    strategies.append({
                        "name": f"{role}_{style}_{c_format}_{history}",
                        "role": role,
                        "style": style,
                        "c_format": c_format,
                        "history": history
                    })
    # 在 run_all_strategies 里 strategies.append(...) 之后追加：
    strategies += [
        {
            "name": "expert_strict_comma",
            "role": "expert",
            "style": "instruction",
            "c_format": "id_first",
            "history": "plain",
            "extra": "comma"
        },
        {
            "name": "expert_strict_json",
            "role": "expert",
            "style": "instruction",
            "c_format": "id_first",
            "history": "plain",
            "extra": "json"
        },
        {
            "name": "expert_cot_hidden",
            "role": "expert",
            "style": "instruction",
            "c_format": "id_first",
            "history": "plain",
            "extra": "cot"
        },
        {
            "name": "expert_format_example",
            "role": "expert",
            "style": "instruction",
            "c_format": "id_first",
            "history": "plain",
            "extra": "example"
        },
    ]

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    results = []
    for strat in strategies:
        # 重复调用取平均
        all_scores = []
        for _ in range(10):
            name, score = await evaluate_strategy(val_data, api_key, strat)
            all_scores.append(score)
        avg_score = sum(all_scores) / len(all_scores)
        results.append((name, avg_score))

    print("\n======= 策略评测结果汇总 =======")
    print("| 策略名称 | NDCG@10 |")
    print("|----------|---------|")
    for name, score in results:
        print(f"| {name} | {score:.4f} |")

    with open("result/zuhe1/prompt_strategy_results.md", "w", encoding="utf-8") as f:
        f.write("| 策略名称 | NDCG@10 |\n")
        f.write("|----------|---------|\n")
        for name, score in results:
            f.write(f"| {name} | {score:.4f} |\n")

    with open("result/zuhe1/prompt_strategy_results.csv", "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["策略名称", "NDCG@10"])
        for name, score in results:
            writer.writerow([name, f"{score:.4f}"])

if __name__ == "__main__":
    API_KEY = "sk-60981d4439e34af29f9b9afef32d6c7e"
    VAL_PATH = "E:/PE_Exam/val.jsonl"
    asyncio.run(run_all_strategies(VAL_PATH, API_KEY))
