import json
import asyncio
import math
import logging
from tqdm import tqdm
from openai import AsyncOpenAI
import re
import csv
import os
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple, Any, Optional


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

def parse_output(text):
    """
    更健壮的解析函数，自动识别中英文逗号、空格、换行分隔的 ID 序列
    """
    logging.info(f"模型输出解析 - 原始输出:\n{text}")

    # 替换所有可能的分隔符为英文逗号
    text = text.replace('\n', ',').replace('，', ',').replace(';', ',').replace('、', ',')
    candidates = re.findall(r'\b\d+\b', text)

    # 去重、保序
    seen = set()
    parsed_list = []
    for id_str in candidates:
        if id_str not in seen:
            seen.add(id_str)
            parsed_list.append(int(id_str))

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



# 设置日志
def setup_logging(log_dir: str = "logs_eval_per_sample", log_name: Optional[str] = None) -> None:
    os.makedirs(log_dir, exist_ok=True)
    if log_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"evaluation_{timestamp}.log"
    log_file = os.path.join(log_dir, log_name)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("日志系统初始化完成")

# 核心评估函数
async def evaluate_detailed(
        val_data: List[Dict[str, Any]],
        api_key: str,
        construct_prompt_func: callable,
        num_trials: int = 10,
        max_concurrency: int = 10,
        k: int = 10,
        result_dir: str = "results_detailed"
) -> Tuple[Dict[str, Any], List[float]]:
    logging.info(f"\n===== 开始评估 ======")
    os.makedirs(result_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrency)
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    num_samples = len(val_data)
    sample_scores = {i: [] for i in range(num_samples)}

    for trial in range(num_trials):
        logging.info(f"开始第 {trial + 1}/{num_trials} 次评估")

        async def worker(sample_idx: int, sample: Dict[str, Any]) -> float:
            async with semaphore:
                try:
                    prompt = construct_prompt_func(sample)
                    output_text = await query_deepseek_async(client, prompt)
                    logging.debug(f"样本 {sample_idx} 输出: {output_text}")
                    predicted_list = parse_output(output_text)
                    if not predicted_list:
                        logging.warning(f"样本 {sample_idx} 输出解析失败: {output_text}")
                        return 0.0
                    ndcg = calculate_ndcg_for_sample(predicted_list, sample["target_item"][0], k)
                    logging.info(f"样本 {sample_idx} (Trial {trial + 1}): NDCG@{k} = {ndcg:.4f}")
                    return ndcg
                except Exception as e:
                    logging.error(f"样本 {sample_idx} (Trial {trial + 1}) 出错: {e}")
                    return 0.0

        tasks = [worker(idx, sample) for idx, sample in enumerate(val_data)]
        trial_scores = [await t for t in
                        tqdm(asyncio.as_completed(tasks), total=num_samples, desc=f"Trial {trial + 1}")]

        for idx, score in enumerate(trial_scores):
            sample_scores[idx].append(score)

    trial_avg_scores = []
    for trial in range(num_trials):
        scores = [sample_scores[idx][trial] for idx in range(num_samples)]
        avg_ndcg = sum(scores) / len(scores)
        trial_avg_scores.append(avg_ndcg)
        logging.info(f"Trial {trial + 1} 平均 NDCG@{k}: {avg_ndcg:.4f}")

    overall_avg_ndcg = sum(trial_avg_scores) / len(trial_avg_scores)
    logging.info(f"总体平均 NDCG@{k}: {overall_avg_ndcg:.4f}")

    sample_stats = {}
    for idx in range(num_samples):
        scores = sample_scores[idx]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        sample_stats[idx] = {"mean_ndcg": mean_score, "std_ndcg": std_score}

    detailed_result = {
        "overall_avg_ndcg": overall_avg_ndcg,
        "trial_avg_scores": trial_avg_scores,
        "sample_stats": sample_stats,
        "num_trials": num_trials,
        "num_samples": num_samples,
        "k": k
    }

    result_file = os.path.join(result_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_result, f, indent=2, ensure_ascii=False)
    logging.info(f"详细结果已保存至: {result_file}")

    csv_file = os.path.join(result_dir, f"sample_scores.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["Sample_ID", "Mean_NDCG", "Std_NDCG"] + [f"Trial_{i + 1}" for i in range(num_trials)]
        writer.writerow(header)
        for idx in range(num_samples):
            row = [idx, f"{sample_stats[idx]['mean_ndcg']:.4f}", f"{sample_stats[idx]['std_ndcg']:.4f}"]
            writer.writerow(row)
    logging.info(f"样本得分CSV已保存至: {csv_file}")

    return detailed_result, trial_avg_scores

# 示例主函数
if __name__ == "__main__":
    setup_logging(log_dir="logs_eval", log_name="detailed_eval.log")

    API_KEY = "sk-60981d4439e34af29f9b9afef32d6c7e"
    VAL_PATH = "E:/PE_Exam/val.jsonl"


    def construct_prompt_sota(d): #0.7206
        """
        构造用于大语言模型的提示词
        参数:
            d (dict): jsonl数据文件的一行，解析成字典后的变量
        返回:
            list: OpenAI API的message格式列表
        示例: [{"role": "system", "content": "系统提示内容"},
               {"role": "user", "content": "用户提示内容"}]
        """
        system_prompt = "请你扮演用户。请根据你最近看过的电影，从候选电影中选择你最可能继续观看的，按兴趣排序。你可以根据类型或主题等维度判断匹配度。"

        user_history = "\n".join([f"- {title} (ID: {mid})" for mid, title in reversed(d['item_list'][-10:])])
        candidate_movies = "\n".join([f"{mid}: {title}" for mid, title in d['candidates']])

        user_prompt = f"我最近看过的电影如下：\n{user_history}\n\n推荐系统为我准备了以下候选电影：\n{candidate_movies}\n\n请根据我的兴趣，从这些候选电影中选出我最想看的，按兴趣程度排序。\n只输出JSON格式的ID列表，例如： [2492, 684, 1893]\n不要输出其他任何解释或文字。"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]



    with open(VAL_PATH, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line.strip()) for line in f]

    results = asyncio.run(evaluate_detailed(
        val_data=val_data,
        api_key=API_KEY,
        construct_prompt_func=construct_prompt_sota,
        num_trials=20,
        max_concurrency=100,
        k=10,
        result_dir="results_detailed"
    ))

    logging.info("评估完成")