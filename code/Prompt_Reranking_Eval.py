import json
import re
import concurrent.futures
import requests
from tqdm import tqdm
import math
from openai import OpenAI
from openai import AsyncOpenAI

import asyncio
import aiohttp
import logging
import os
from datetime import datetime


# 设置日志配置
def setup_logging(log_dir="logs"):
    """
    设置日志记录配置，将日志保存到文件并输出到控制台。
    参数:
        log_dir (str): 日志文件存储目录
    返回:
        None
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"evaluation_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            # logging.StreamHandler()
        ]
    )
    logging.info("日志系统初始化完成")

# -------------- 核心函数 ---------------

def construct_prompt(d):
    """
    构造用于大语言模型的提示词
    参数:
        d (dict): jsonl数据文件的一行，解析成字典后的变量
    返回:
        list: OpenAI API的message格式列表，允许设计多轮对话式的prompt
    示例: [{"role": "system", "content": "系统提示内容"},
    {"role": "user", "content": "用户提示内容"}]
    """
    # 实现提示词构造逻辑

    system_prompt = "你是一名电影推荐专家，根据用户的历史观影记录，对候选电影进行重排，越可能被用户观看的电影排得越靠前。"

    user_history = "\n".join([f"- {movie[1]}" for movie in d['item_list'][-10:]])
    candidate_movies = "\n".join([f"{movie[0]}: {movie[1]}" for movie in d['candidates']])

    user_prompt = f"用户最近观看的电影：\n{user_history}\n\n请根据用户的兴趣，对以下候选电影进行排序（输出电影ID列表，最可能观看的电影在最前）：\n{candidate_movies}\n\n直接输出电影ID列表，不要额外的解释或文字。"


    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 记录提示词内容
    logging.info(f"构建提示词 - 用户历史记录:\n{user_history}")
    logging.info(f"构建提示词 - 候选电影列表:\n{candidate_movies}")
    logging.info(f"构建提示词 - 完整用户提示:\n{user_prompt}")

    return prompt_messages


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

def construct_prompt_v2(d):
    system_content = """你是一个电影推荐专家，需要根据用户观影历史对候选电影进行重排序。分析用户兴趣时请重点关注近期观看的电影。输出必须严格按可能性从高到低排列所有候选ID，用逗号分隔，不要任何其他内容。"""

    # 历史记录反向展示（最后观看的在下）
    reversed_history = reversed(d["item_list"])
    history = "\n".join([f"- {title} (ID: {id})" for id, title in reversed_history])

    # 候选列表带ID防歧义
    candidates = "\n".join([f"- {title} (ID: {id})" for id, title in d["candidates"]])

    user_content = f"""用户观影历史（按时间排序，最后观看的在最下方）：
    {history}

    候选电影列表（需重新排序）：
    {candidates}

    请按以下要求处理：
    1. 分析用户观影偏好演变，特别注意最近观看的电影
    2. 比较候选电影与用户兴趣的匹配程度
    3. 输出必须包含所有候选ID（20个）
    4. 用英文逗号分隔ID，不要换行，示例：1234,5678,9012"""

    # 日志记录保持兼容
    logging.info(f"构建提示词 - 用户历史记录:\n{history}")
    logging.info(f"构建提示词 - 候选电影列表:\n{candidates}")

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

def parse_output_v2(text):
    """
    解析大语言模型的输出文本，提取推荐重排列表
    参数:
        text (str): 大语言模型在设计prompt下的输出文本
    返回:
        list: 解析出的电影ID列表
    """
    # 增强容错性：兼容中英文逗号、空格分隔
    ids = re.findall(r'\b\d+\b', text.replace('，', ',').replace(' ', ','))

    # 去重处理并保持顺序
    seen = set()
    unique_ids = []
    for id_str in ids:
        if id_str not in seen:
            seen.add(id_str)
            unique_ids.append(int(id_str))

    logging.info(f"模型输出解析 - 解析后电影ID列表: {unique_ids}")
    return unique_ids

# -------------- 高效评测函数 ---------------
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


# 同步请求函数
def query_deepseek(client, prompt_messages):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=prompt_messages,
            temperature=0,
            max_tokens=8192
        )
        output_text = response.choices[0].message.content
        logging.info(f"DeepSeek API 同步请求 - 模型输出:\n{output_text}")
        return output_text
    except Exception as e:
        logging.error(f"DeepSeek API 同步请求失败: {e}")
        raise

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

# 单线程评测函数
def evaluate(val_data, api_key):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    ndcgs = []
    logging.info("开始单线程评测")
    for idx, d in enumerate(tqdm(val_data)):
        logging.info(f"处理样本 {idx+1}/{len(val_data)}")
        prompt = construct_prompt(d)
        output_text = query_deepseek(client, prompt)
        predicted_list = parse_output(output_text)
        ndcg = calculate_ndcg_for_sample(predicted_list, d["target_item"][0])
        ndcgs.append(ndcg)
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0
    logging.info(f"单线程评测完成 - NDCG@10: {ndcgs}")
    logging.info(f"单线程评测完成 - 平均 NDCG@10: {avg_ndcg}")
    return avg_ndcg


# 高性能并发评测函数
def evaluate_parallel(val_data, api_key, max_workers=10):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    ndcgs = []
    logging.info(f"开始并发评测，最大线程数: {max_workers}")

    def worker(d):
        prompt = construct_prompt(d)
        output_text = query_deepseek(client, prompt)
        predicted_list = parse_output(output_text)
        return calculate_ndcg_for_sample(predicted_list, d["target_item"][0])

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(worker, val_data), total=len(val_data)))

    ndcgs = results
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0
    logging.info(f"并发评测完成 - 平均 NDCG@10: {avg_ndcg}")
    return avg_ndcg

# 更高效的异步评测函数
async def evaluate_async(val_data, api_key, max_concurrency=10, construct_prompt=None, parse_output=None):
    semaphore = asyncio.Semaphore(max_concurrency)
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    logging.info(f"开始异步评测，最大并发数: {max_concurrency}")

    async def worker(d):
        async with semaphore:
            prompt = construct_prompt(d)
            output_text = await query_deepseek_async(client, prompt)
            predicted_list = parse_output(output_text)
            return calculate_ndcg_for_sample(predicted_list, d["target_item"][0])

    tasks = [worker(d) for d in val_data]
    ndcgs = [await f for f in tqdm(asyncio.as_completed(tasks), total=len(tasks))]
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0
    logging.info(f"异步评测完成 -  NDCG@10: {ndcgs}")
    logging.info(f"异步评测完成 - 平均 NDCG@10: {avg_ndcg}")
    return avg_ndcg


# -------------- 数据加载示例 ---------------

if __name__ == "__main__":

    # 初始化日志系统
    setup_logging()

    DEEPSEEK_API_KEY = "sk-60981d4439e34af29f9b9afef32d6c7e"
    DEEPSEEK_Client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    response = DEEPSEEK_Client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )

    print(response.choices[0].message.content)

    # with open('E:\PE_Exam\\val.jsonl', 'r', encoding='utf-8') as f:
    #     val_data = [json.loads(line) for line in f]
    #
    # # 单线程评测
    # ndcg_score = evaluate(val_data, api_key=DEEPSEEK_API_KEY)
    # print(f"单线程评测 NDCG@10: {ndcg_score}")
    #
    # # 高性能并发评测
    # ndcg_score_parallel = evaluate_parallel(val_data, api_key=DEEPSEEK_API_KEY, max_workers=20)
    # print(f"并发评测 NDCG@10: {ndcg_score_parallel}")
    #
    # # 运行异步评测
    # ndcg_score_async = asyncio.run(evaluate_async(val_data, api_key=DEEPSEEK_API_KEY, max_concurrency=20))
    # print(f"异步评测 NDCG@10: {ndcg_score_async}")


    # 加载验证数据
    data_path = 'E:\\PE_Exam\\val.jsonl'
    logging.info(f"加载验证数据 - 路径: {data_path}")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            val_data = [json.loads(line) for line in f]
        logging.info(f"验证数据加载成功 - 样本数: {len(val_data)}")
    except Exception as e:
        logging.error(f"验证数据加载失败: {e}")
        val_data = []

    # if val_data:
    #     # 单线程评测
    #     logging.info("启动单线程评测...")
    #     ndcg_score = evaluate(val_data, api_key=DEEPSEEK_API_KEY)
    #     print(f"单线程评测 NDCG@10: {ndcg_score}")
    #
    #     # 高性能并发评测
    #     logging.info("启动并发评测...")
    #     ndcg_score_parallel = evaluate_parallel(val_data, api_key=DEEPSEEK_API_KEY, max_workers=20)
    #     print(f"并发评测 NDCG@10: {ndcg_score_parallel}")
    #
    #     # 运行异步评测
    #     logging.info("启动异步评测...")
    #     ndcg_score_async = asyncio.run(evaluate_async(val_data, api_key=DEEPSEEK_API_KEY, max_concurrency=20))
    #     print(f"异步评测 NDCG@10: {ndcg_score_async}")
    # else:
    #     logging.warning("无验证数据，跳过评测")
    #
    # print(f"单线程评测 NDCG@10: {ndcg_score}")
    # print(f"并发评测 NDCG@10: {ndcg_score_parallel}")
    # print(f"异步评测 NDCG@10: {ndcg_score_async}")

    ndcg_score_async_list = []
    for _ in range(10):
        ndcg_score_async = asyncio.run(evaluate_async(val_data, api_key=DEEPSEEK_API_KEY, max_concurrency=20, construct_prompt=construct_prompt_v2, parse_output=parse_output_v2))
        ndcg_score_async_list.append(ndcg_score_async)

    print(f"Avg异步评测 NDCG@10: {sum(ndcg_score_async_list) / len(ndcg_score_async_list)}")
