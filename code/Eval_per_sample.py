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

from Prompt_Reranking_Eval import (
    calculate_ndcg_for_sample,
    parse_output,
    query_deepseek_async,
)

from Prompt_Strategy_Eval import construct_prompt_variant


# 设置日志
def setup_logging(log_dir: str = "logs_eval_per_sample", log_name: Optional[str] = None) -> None:
    """
    设置日志记录配置，将日志保存到文件。
    参数:
        log_dir (str): 日志文件存储目录
        log_name (str, optional): 日志文件名，如果未提供则使用时间戳
    返回:
        None
    """
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
            # logging.StreamHandler()  # 同时输出到控制台以便调试
        ]
    )
    logging.info("日志系统初始化完成")



# 核心评估函数
async def evaluate_detailed(
        val_data: List[Dict[str, Any]],
        api_key: str,
        strategy: Dict[str, str],
        construct_prompt_func: callable,
        num_trials: int = 10,
        max_concurrency: int = 10,
        k: int = 10,
        result_dir: str = "results_detailed"
) -> Tuple[Dict[str, Any], List[float]]:
    """
    详细评估函数，对每个样本进行多次评估，记录每个样本的得分并分析策略表现。
    参数:
        val_data (List[Dict[str, Any]]): 验证数据集
        api_key (str): API密钥
        strategy (Dict[str, str]): 使用的提示策略
        construct_prompt_func (callable): 构造提示词的函数
        num_trials (int): 每个样本的评估次数
        max_concurrency (int): 最大并发数
        k (int): 计算NDCG时的前k项
        result_dir (str): 结果保存目录
    返回:
        Tuple[Dict[str, Any], List[float]]: 详细结果字典和每次评估的平均NDCG列表
    """
    logging.info(f"\n===== 开始评估策略: {strategy['name']} ======")
    os.makedirs(result_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrency)
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    num_samples = len(val_data)
    sample_scores = {i: [] for i in range(num_samples)}  # 每个样本的多次评估得分

    # 对每个样本进行num_trials次评估
    for trial in range(num_trials):
        logging.info(f"开始第 {trial + 1}/{num_trials} 次评估")

        async def worker(sample_idx: int, sample: Dict[str, Any]) -> float:
            async with semaphore:
                try:
                    prompt = construct_prompt_func(sample, strategy)
                    output_text = await query_deepseek_async(client, prompt)
                    predicted_list = parse_output(output_text)
                    ndcg = calculate_ndcg_for_sample(predicted_list, sample["target_item"][0], k)
                    logging.info(f"样本 {sample_idx} (Trial {trial + 1}): NDCG@{k} = {ndcg:.4f}")
                    return ndcg
                except Exception as e:
                    logging.warning(f"样本 {sample_idx} (Trial {trial + 1}) 出错: {e}")
                    return 0.0

        # 并发处理所有样本
        tasks = [worker(idx, sample) for idx, sample in enumerate(val_data)]
        trial_scores = [await t for t in
                        tqdm(asyncio.as_completed(tasks), total=num_samples, desc=f"Trial {trial + 1}")]

        # 记录每个样本的得分
        for idx, score in enumerate(trial_scores):
            sample_scores[idx].append(score)

    # 计算统计量
    trial_avg_scores = []
    for trial in range(num_trials):
        scores = [sample_scores[idx][trial] for idx in range(num_samples)]
        avg_ndcg = sum(scores) / len(scores)
        trial_avg_scores.append(avg_ndcg)
        logging.info(f"Trial {trial + 1} 平均 NDCG@{k}: {avg_ndcg:.4f}")

    overall_avg_ndcg = sum(trial_avg_scores) / len(trial_avg_scores)
    logging.info(f"策略 {strategy['name']} 总体平均 NDCG@{k}: {overall_avg_ndcg:.4f}")

    # 计算每个样本的平均得分和标准差
    sample_stats = {}
    for idx in range(num_samples):
        scores = sample_scores[idx]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        sample_stats[idx] = {
            "mean_ndcg": mean_score,
            "std_ndcg": std_score,
            # "scores": scores  ################################################################################################
        }
        logging.info(f"样本 {idx}: 平均 NDCG@{k} = {mean_score:.4f}, 标准差 = {std_score:.4f}")

    # 保存详细结果
    detailed_result = {
        "strategy_name": strategy['name'],
        "overall_avg_ndcg": overall_avg_ndcg,
        "trial_avg_scores": trial_avg_scores,
        "sample_stats": sample_stats,
        "num_trials": num_trials,
        "num_samples": num_samples,
        "k": k
    }

    # 保存为JSON
    result_file = os.path.join(result_dir, f"eval_{strategy['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_result, f, indent=2, ensure_ascii=False)
    logging.info(f"详细结果已保存至: {result_file}")

    # 保存为CSV（便于查看每个样本的得分）
    csv_file = os.path.join(result_dir, f"sample_scores_{strategy['name']}.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["Sample_ID", "Mean_NDCG", "Std_NDCG"] + [f"Trial_{i + 1}" for i in range(num_trials)]
        writer.writerow(header)
        for idx in range(num_samples):
            row = [idx, f"{sample_stats[idx]['mean_ndcg']:.4f}", f"{sample_stats[idx]['std_ndcg']:.4f}"]
            # + \
                  # [f"{score:.4f}" for score in sample_stats[idx]['scores']] ##################################################
            writer.writerow(row)
    logging.info(f"样本得分CSV已保存至: {csv_file}")

    return detailed_result, trial_avg_scores


# 运行多个策略的函数
async def run_multiple_strategies(
        val_data_path: str,
        api_key: str,
        strategies: List[Dict[str, str]],
        construct_prompt_func: callable,
        num_trials: int = 10,
        max_concurrency: int = 10,
        k: int = 10,
        result_dir: str = "results_detailed"
) -> Dict[str, Any]:
    """
    对多个策略进行详细评估，比较每个策略的表现。
    参数:
        val_data_path (str): 验证数据文件路径
        api_key (str): API密钥
        strategies (List[Dict[str, str]]): 策略列表
        construct_prompt_func (callable): 构造提示词的函数
        num_trials (int): 每个样本的评估次数
        max_concurrency (int): 最大并发数
        k (int): 计算NDCG时的前k项
        result_dir (str): 结果保存目录
    返回:
        Dict[str, Any]: 所有策略的评估结果
    """
    # 加载数据
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line.strip()) for line in f]
    logging.info(f"验证数据加载成功 - 样本数: {len(val_data)}")

    # 评估每个策略
    all_results = {}
    summary = []
    for strategy in strategies:
        logging.info(f"\n===== 评估策略: {strategy['name']} ======")
        detailed_result, trial_scores = await evaluate_detailed(
            val_data, api_key, strategy, construct_prompt_func,
            num_trials, max_concurrency, k, result_dir
        )
        all_results[strategy['name']] = detailed_result
        summary.append((strategy['name'], detailed_result['overall_avg_ndcg']))

    # 打印汇总结果
    print("\n======= 策略评测结果汇总 =======")
    print("| 策略名称 | NDCG@10 |")
    print("|----------|---------|")
    for name, score in summary:
        print(f"| {name} | {score:.4f} |")

    # 保存汇总结果
    summary_file = os.path.join(result_dir, "strategy_summary.md")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("| 策略名称 | NDCG@10 |\n")
        f.write("|----------|---------|\n")
        for name, score in summary:
            f.write(f"| {name} | {score:.4f} |\n")
    logging.info(f"策略汇总结果已保存至: {summary_file}")

    return all_results


# 示例主函数
if __name__ == "__main__":
    raise ValueError
    # 初始化日志系统
    setup_logging(log_dir="logs_eval", log_name="detailed_eval.log")

    # API和数据路径
    API_KEY = "sk-60981d4439e34af29f9b9afef32d6c7e"
    VAL_PATH = "E:/PE_Exam/val.jsonl"

    # 定义策略（从File 2中提取部分策略作为示例）
    strategies = [
        {
            "name": "expert_instruction_id_first_plain",
            "role": "expert",
            "style": "instruction",
            "c_format": "id_first",
            "history": "plain"
        },
        # {
        #     "name": "analyst_question_name_first_timestamped",
        #     "role": "analyst",
        #     "style": "question",
        #     "c_format": "name_first",
        #     "history": "timestamped"
        # },
        # {
        #     "name": "expert_strict_comma",
        #     "role": "expert",
        #     "style": "instruction",
        #     "c_format": "id_first",
        #     "history": "plain",
        #     "extra": "comma"
        # }
    ]

    # 构造提示词函数（从File 2中提取，需根据实际代码调整）

    def construct_prompt_best_058(d, **kwargs):
        system_prompt = (
            "你是一名电影推荐专家，擅长根据用户的历史观影记录分析其兴趣偏好，并对候选电影进行精准排序。你的目标是预测用户下一步最可能观看的电影，并将候选电影按可能性从高到低重排。"
        )

        # 取最近10部电影
        history_movies = "\n".join([f"- {movie[0]}: {movie[1]}" for movie in d['item_list'][-10:]])
        candidate_movies = "\n".join([f"{movie[0]}: {movie[1]}" for movie in d['candidates']])

        user_prompt = (
            f"以下是用户按时间顺序观看的最近10部电影（越靠后越近期）：\n"
            f"{history_movies}\n\n"
            f"请分析用户历史观影记录中的模式和趋势，例如电影类型（如科幻、动作、喜剧）、主题（如冒险、爱情）或风格偏好，并基于这些分析对以下候选电影进行排序，越可能被用户观看的电影排在越前面：\n"
            f"{candidate_movies}\n\n"
            f"注意电影标题中的关键词，例如‘科幻’、‘动作’、‘爱情’等，并考虑这些关键词与用户历史观影记录的匹配程度。\n"
            f"直接以JSON格式输出排序后的电影ID列表，例如：{{\"recommended_movies\": [1893, 2492, 684]}}，不要包含任何额外的解释或文字。"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


    def construct_prompt(d, strategy, **kwargs):
        system_prompt = (
            "你是一名电影推荐专家，擅长从用户的历史观影记录中分析其兴趣模式和偏好。你的任务是根据用户最近的观影记录，精准排序候选电影，确保推荐结果高度契合用户兴趣。"
        )

        # 取最近5部电影，按时间顺序展示
        user_history = "\n".join([f"- {movie[0]}: {movie[1]}" for movie in d['item_list'][-5:]])

        # 候选电影展示
        candidate_movies = "\n".join([f"{movie[0]}: {movie[1]}" for movie in d['candidates']])

        user_prompt = (
            f"请根据以下用户最近观看的5部电影（按时间顺序，越靠后越近期），完成以下任务：\n"
            f"{user_history}\n\n"
            f"**步骤 1：分析用户偏好**\n"
            f"- 分析电影的类型（如科幻、动作、喜剧、恐怖、爱情、悬疑等）、主题（如冒险、家庭、犯罪、历史等）或风格（如幽默、紧张、感人等）。\n"
            f"- 特别关注最近2-3部电影，总结用户的当前兴趣模式。\n\n"
            f"**步骤 2：排序候选电影**\n"
            f"根据你的分析，对以下候选电影进行排序，确保最符合用户当前兴趣的电影排在前面，同时考虑推荐的多样性和新鲜度：\n"
            f"{candidate_movies}\n\n"
            f"**重要提示**：\n"
            f"- 从电影标题中提取关键词（如‘科幻’、‘动作’、‘爱情’等），与用户偏好匹配。\n"
            f"- 优先考虑近期观影记录，因其更能反映用户当前需求。\n"
            f"- 确保推荐的电影在类型和主题上与用户偏好高度契合，同时避免过于相似的推荐。\n"
            f"- 严格以JSON格式输出排序后的电影ID列表，例如：{{\"recommended_movies\": [1893, 2492, 684]}}，不包含任何解释或多余文字。"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


    def construct_prompt_naive(d):
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

    # 运行评估
    results = asyncio.run(run_multiple_strategies(
        val_data_path=VAL_PATH,
        api_key=API_KEY,
        strategies=strategies,
        construct_prompt_func=construct_prompt,
        num_trials=10,
        max_concurrency=10,
        k=10,
        result_dir="results_detailed"
    ))
    logging.info("所有策略评估完成")
