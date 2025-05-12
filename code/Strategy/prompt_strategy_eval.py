import json
import asyncio
import math
import logging
from tqdm import tqdm
from openai import AsyncOpenAI
import re
import csv

def calculate_ndcg_for_sample(predicted_list, ground_truth_item, k=10):
    # relevance = [1 if item_id == ground_truth_item else 0 for item_id in predicted_list[:k]]
    # dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
    # return dcg  # 因为IDCG始终为1，所以省略
    relevance = [1 if item_id == ground_truth_item else 0 for item_id in predicted_list[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
    idcg = 1 / math.log2(1 + 1)  # 理想情况：相关项在第一位
    ndcg = dcg / idcg if idcg > 0 else 0
    logging.info(f"NDCG计算 - 预测列表前{k}项: {predicted_list[:k]}, 真实目标: {ground_truth_item}, NDCG@{k}: {ndcg}")
    return ndcg



def parse_output(text):
    """
    解析大语言模型的输出文本，提取推荐重排列表
    参数:
        text (str): 大语言模型在设计prompt下的输出文本
    返回:
    list: 从输出文本解析出的电影ID列表（python列表格式，列表的每个元素是整数，表示编
    号），表示重排后的推荐顺序
    示例: [1893, 3148, 111, ...]
    """
    parsed_list = list(map(int, re.findall(r'\d+', text)))
    logging.info(f"模型输出解析 - 原始输出:\n{text}")
    logging.info(f"模型输出解析 - 解析后电影ID列表: {parsed_list}")
    return parsed_list


# 异步请求函数
async def query_deepseek_async(client, prompt_messages):
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=prompt_messages,
            temperature=0,
            max_tokens=8192
        )
        output_text = response.choices[0].message.content
        logging.info(f"DeepSeek API 异步请求 - 模型输出:\n{output_text}")
        return output_text
    except Exception as e:
        logging.error(f"DeepSeek API 异步请求失败: {e}")
        raise


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



def construct_prompt_variant(d, strategy):
    """根据策略组合生成 Prompt 消息列表，融合论文中的提示变体"""
    system_roles = {
        "expert": "你是一名电影推荐专家，目标是根据用户历史预测他们最可能观看的电影。",
        "analyst": "你是一名数据分析师，请基于用户历史行为重排候选电影。",
        # 融合论文中的 Role Playing 概念
        "rankgpt": "你是 RankGPT，一个智能助手，可以根据用户历史记录对电影进行相关性排名。"
    }

    styles = {
        "instruction": "请根据用户的兴趣偏好，对下列候选电影重排，最可能观看的放最前，仅返回电影ID列表：",
        "question": "用户接下来最有可能观看哪些电影？请从以下候选中排序，只返回ID列表：",
        # 融合论文中的 Task Instruction 变体
        "rank_query": "根据用户历史记录，排名以下 {num} 部电影，按与用户偏好的相关性从高到低排序。"
    }

    candidates_fmt = {
        "id_first": lambda c: "\n".join([f"{i[0]}: {i[1]}" for i in c]),
        "name_first": lambda c: "\n".join([f"{i[1]} (ID:{i[0]})" for i in c]),
        # 融合论文中的 Evidence Ordering (QF/PF)
        "id_first_indexed": lambda c: "\n".join([f"[{idx+1}] {i[0]}: {i[1]}" for idx, i in enumerate(c)])
    }

    history_fmt = {
        "timestamped": lambda h: "\n".join([f"[{idx+1}] {m[1]}" for idx, m in enumerate(h)]),
        "plain": lambda h: "\n".join([f"- {m[1]}" for m in h])
    }

    # ---- 组装各部分 ----
    system_prompt = system_roles.get(strategy.get('role', 'expert'), system_roles['expert'])
    user_instruction = styles.get(strategy.get('style', 'instruction'), styles['instruction'])

    extra = strategy.get("extra", "")
    if extra == "comma":
        user_instruction += "\n\n输出格式：用英文逗号分隔一行列出电影ID，不要有空格或额外文字。"
    elif extra == "json":
        user_instruction += "\n\n输出格式：仅返回 JSON 数组，如 [2492,684,…]。"
    elif extra == "cot":
        user_instruction += "\n\n先在心里思考，不要输出思考过程，仅给出最终 ID 列表。"
    # 融合论文中的 Output Type 变体
    elif extra == "rank_format":
        user_instruction += f"\n\n输出格式：按降序排列电影标识符，格式为 [] > [], 例如 [1] > [2]，只输出排名结果，不解释或添加任何其他文字。"

    history_text = history_fmt[strategy.get('history', 'plain')](d['item_list'][-10:])
    candidates_text = candidates_fmt.get(strategy.get('c_format', 'id_first'), candidates_fmt['id_first'])(d['candidates'])

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
    # 融合论文中的 Tone Words
    elif strategy.get("tone", "") == "please":
        prompt_body += "\n\n请确保排名准确反映用户偏好。"
    elif strategy.get("tone", "") == "must":
        prompt_body += "\n\n必须严格按照用户偏好排序，不得随意排列。"

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
    for role in ['expert', 'analyst', 'rankgpt']:  # 增加论文中的 Role Playing
        for style in ['instruction', 'question', 'rank_query']:  # 增加论文中的 Task Instruction
            for c_format in ['id_first', 'name_first', 'id_first_indexed']:  # 增加论文中的 Evidence Ordering
                for history in ['plain', 'timestamped']:
                    strategies.append({
                        "name": f"{role}_{style}_{c_format}_{history}",
                        "role": role,
                        "style": style,
                        "c_format": c_format,
                        "history": history
                    })
    # 增加论文中的 Tone Words 和 Output Type 变体
    strategies += [
        {
            "name": "expert_strict_comma_tone_please",
            "role": "expert",
            "style": "instruction",
            "c_format": "id_first",
            "history": "plain",
            "extra": "comma",
            "tone": "please"
        },
        {
            "name": "expert_strict_json_tone_must",
            "role": "expert",
            "style": "instruction",
            "c_format": "id_first",
            "history": "plain",
            "extra": "json",
            "tone": "must"
        },
        {
            "name": "rankgpt_rank_format",
            "role": "rankgpt",
            "style": "rank_query",
            "c_format": "id_first_indexed",
            "history": "plain",
            "extra": "rank_format"
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
    API_KEY = "sk-cb2bd9b9470c41ed9daca64aa8253319"
    VAL_PATH = "E:/PE_Exam/val.jsonl"
    asyncio.run(run_all_strategies(VAL_PATH, API_KEY))
