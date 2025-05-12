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

# 计算NDCG的函数，保持不变
def calculate_ndcg_for_sample(predicted_list, ground_truth_item, k=10):
    relevance = [1 if item_id == ground_truth_item else 0 for item_id in predicted_list[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
    idcg = 1 / math.log2(1 + 1)  # 理想情况：相关项在第一位
    ndcg = dcg / idcg if idcg > 0 else 0
    logging.info(f"NDCG计算 - 预测列表前{k}项: {predicted_list[:k]}, 真实目标: {ground_truth_item}, NDCG@{k}: {ndcg}")
    return ndcg

# 解析输出的函数，保持不变
def parse_output(text):
    """
    更健壮的解析函数，自动识别中英文逗号、空格、换行分隔的ID序列
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

# 异步请求函数，保持不变
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

# 新增：过滤数据集以排除指定样本
def filter_dataset(data: List[Dict[str, Any]], exclude_indices: List[int] = None) -> List[Dict[str, Any]]:
    """
    过滤数据集，排除指定索引的样本。
    参数:
        data: 完整的数据集
        exclude_indices: 要排除的样本索引列表（默认为None，表示不排除任何样本）
    返回:
        filtered_data: 过滤后的数据集
    """
    if exclude_indices is None:
        exclude_indices = []
    filtered_data = [sample for idx, sample in enumerate(data) if idx not in exclude_indices]
    logging.info(f"数据集过滤 - 总样本: {len(data)}, 排除样本: {len(exclude_indices)}, 剩余样本: {len(filtered_data)}")
    return filtered_data

# 核心评估函数，添加exclude_indices参数
async def evaluate_detailed(
        val_data: List[Dict[str, Any]],
        api_key: str,
        construct_prompt_func: callable,
        num_trials: int = 10,
        max_concurrency: int = 10,
        k: int = 10,
        result_dir: str = "results_detailed",
        exclude_indices: List[int] = None
) -> Tuple[Dict[str, Any], List[float]]:
    logging.info(f"\n===== 开始评估 ======")
    os.makedirs(result_dir, exist_ok=True)

    # 过滤数据集，排除指定样本
    filtered_data = filter_dataset(val_data, exclude_indices)
    if not filtered_data:
        logging.error("过滤后数据集为空，无法进行评估")
        return {"overall_avg_ndcg": 0.0}, []

    semaphore = asyncio.Semaphore(max_concurrency)
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    num_samples = len(filtered_data)
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

        tasks = [worker(idx, sample) for idx, sample in enumerate(filtered_data)]
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

# 设置日志，保持不变
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

# 构造few-shot提示词，使用固定定义的few-shot样例
def construct_prompt_with_few_shot(d: Dict[str, Any], few_shot_examples: List[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """
    构造包含few-shot示例的提示词，保持简洁高效，使用固定传入的few-shot样例。
    参数:
        d: 当前测试样本
        few_shot_examples: 固定定义的few-shot样例列表（默认为None，将使用下方定义的默认样例）
    返回:
        OpenAI API的message格式列表
    """
    system_prompt = "你是用户本人。请根据最近看过的电影，从候选电影中选择最可能继续观看的，按兴趣排序。可以参考类型或主题进行判断。"

    # 默认固定few-shot样例（可手动修改或通过大模型交互生成）
    # 如果未传入few_shot_examples，则使用以下默认样例
    if few_shot_examples is None:
        few_shot_examples = [
            {
                "history": ["《星际穿越》", "《银翼杀手2049》", "《月球》"],
                "candidates": ["123", "456", "789"],
                "sorted_choice": ["456", "123", "789"]  # 假设456是科幻电影，优先推荐
            },
            {
                "history": ["《宿醉》", "《超级家庭》", "《冒牌天神》"],
                "candidates": ["321", "654", "987"],
                "sorted_choice": ["654", "321", "987"]  # 假设654是喜剧电影，优先推荐
            }
        ]

    # 构建few-shot示例部分，限制为2-3个样例，保持简洁
    few_shot_text = ""
    for idx, example in enumerate(few_shot_examples[:3]):  # 限制为最多3个样例
        history_movies = "\n".join([f"- {movie}" for movie in example["history"]])
        candidate_movies = ", ".join(example["candidates"])
        sorted_choice = ", ".join(example["sorted_choice"])
        few_shot_text += (
            f"示例 {idx + 1}:\n"
            f"我的历史观影（最近3部）：\n"
            f"{history_movies}\n"
            f"候选电影ID（部分）：{candidate_movies}\n"
            f"我的排序选择：{sorted_choice}\n\n"
        )

    # 当前测试样本的提示
    history_movies = "\n".join([f"- {movie[1]}" for movie in d['item_list'][-5:]])  # 简化，只展示最近5部
    candidate_movies = "\n".join([f"{mid}: {title}" for mid, title in d['candidates']])
    user_prompt = (
        f"以下是几个示例，展示如何根据我的观影历史选择感兴趣的电影：\n\n"
        f"{few_shot_text}"
        f"现在，请根据我的观影历史进行排序：\n"
        f"我的历史观影（最近5部）：\n"
        f"{history_movies}\n\n"
        f"候选电影：\n"
        f"{candidate_movies}\n\n"
        f"请选出我最想看的电影，按兴趣排序。只输出ID列表，用逗号分隔，例如：1893, 2492, 684\n"
        f"不要输出其他任何解释或文字。"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# 修改主函数，支持few-shot样例传入和样本遮蔽
if __name__ == "__main__":
    setup_logging(log_dir="logs_eval", log_name="few_shot_eval.log")

    API_KEY = "sk-60981d4439e34af29f9b9afef32d6c7e"
    VAL_PATH = "E:/PE_Exam/val.jsonl"

    # 加载数据
    with open(VAL_PATH, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line.strip()) for line in f]

    # 定义要排除的样本索引（可手动修改，例如排除表现较差的样本0和9）
    exclude_indices = []  # 示例：排除索引为0和9的样本，可根据需要调整

    # 定义自定义的few-shot样例（可手动修改或通过大模型交互生成）
    custom_few_shot_examples = [
        {
            "history": ["Star Trek: Insurrection", "Independence Day (ID4)", "X-Files: Fight the Future"],
            "candidates": ["3699", "2011", "2467", "2482", "2703"],
            "sorted_choice": ["2011", "3699", "2467"]  # 假设2011是科幻电影（如Back to the Future Part II），优先推荐
        },
        {
            "history": ["Monty Python and the Holy Grail", "Young Frankenstein", "Blazing Saddles"],
            "candidates": ["945", "1194", "785", "194", "2022"],
            "sorted_choice": ["1194", "785", "945"]  # 假设1194是喜剧电影（如Up in Smoke），优先推荐
        },
        {
            "history": ["American History X", "Dead Man Walking", "In the Name of the Father"],
            "candidates": ["1458", "534", "3247", "3183", "2791"],
            "sorted_choice": ["534", "1458", "3183"]  # 假设534是剧情电影（如Shadowlands），优先推荐
        }
    ]


    # 使用自定义few-shot样例构造提示函数
    def construct_prompt_func(d):
        return construct_prompt_with_few_shot(d, custom_few_shot_examples)

    # 运行评估，支持样本遮蔽
    results = asyncio.run(evaluate_detailed(
        val_data=val_data,
        api_key=API_KEY,
        construct_prompt_func=construct_prompt_func,
        num_trials=1,
        max_concurrency=100,
        k=10,
        result_dir="results_few_shot_custom",
        exclude_indices=exclude_indices  # 传入要排除的样本索引
    ))

    logging.info("Few-shot评估完成（使用自定义样例和样本遮蔽）")
