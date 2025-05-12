import json
import math
import logging
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from openai import AsyncOpenAI
from tqdm import tqdm
import asyncio
import csv


# 设置日志
def setup_logging(log_dir: str = "logs_llm4rerank", log_name: Optional[str] = None) -> None:
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
        log_name = f"llm4rerank_{timestamp}.log"
    log_file = os.path.join(log_dir, log_name)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("LLM4Rerank 日志系统初始化完成")


# 评估指标实现
def calculate_hr(predicted_list: List[int], ground_truth_item: int, k: int = 10) -> float:
    """
    计算 Hit Ratio (HR@k)。
    参数:
        predicted_list (List[int]): 预测的推荐列表
        ground_truth_item (int): 真实目标项的ID
        k (int): 前k项
    返回:
        float: HR@k 值（1.0 表示命中，0.0 表示未命中）
    """
    return 1.0 if ground_truth_item in predicted_list[:k] else 0.0


def calculate_ndcg(predicted_list: List[int], ground_truth_item: int, k: int = 10) -> float:
    """
    计算 Normalized Discounted Cumulative Gain (NDCG@k)。
    参数:
        predicted_list (List[int]): 预测的推荐列表
        ground_truth_item (int): 真实目标项的ID
        k (int): 前k项
    返回:
        float: NDCG@k 值
    """
    relevance = [1 if item_id == ground_truth_item else 0 for item_id in predicted_list[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
    idcg = 1 / math.log2(1 + 1)  # 理想情况：相关项在第一位
    return dcg / idcg if idcg > 0 else 0


def calculate_alpha_ndcg(predicted_list: List[int], item_groups: Dict[int, str], k: int = 10,
                         alpha: float = 0.5) -> float:
    """
    计算 α-NDCG，用于评估多样性。
    参数:
        predicted_list (List[int]): 预测的推荐列表
        item_groups (Dict[int, str]): 物品ID到分组（如 genre）的映射
        k (int): 前k项
        alpha (float): 多样性权重参数
    返回:
        float: α-NDCG 值
    """
    if not item_groups or len(predicted_list) < k:
        return 0.0
    seen_groups = set()
    gain = 0.0
    for i, item_id in enumerate(predicted_list[:k]):
        group = item_groups.get(item_id, "unknown")
        if group not in seen_groups:
            gain += (1 - alpha) / math.log2(i + 2)
            seen_groups.add(group)
        else:
            gain += alpha / math.log2(i + 2)
    idcg = sum((1 - alpha) / math.log2(i + 2) for i in range(min(len(set(item_groups.values())), k)))
    return gain / idcg if idcg > 0 else 0.0


def calculate_mad(predicted_list: List[int], group_mapping: Dict[int, int], k: int = 10) -> float:
    """
    计算 Mean Absolute Difference (MAD)，用于评估公平性。
    参数:
        predicted_list (List[int]): 预测的推荐列表
        group_mapping (Dict[int, int]): 物品ID到分组（0或1）的映射
        k (int): 前k项
    返回:
        float: MAD 值
    """
    if not group_mapping or not predicted_list:
        return 0.0
    # 分配线性分数，从1到0（根据论文描述）
    scores = {item_id: 1.0 - (i / len(predicted_list)) for i, item_id in enumerate(predicted_list[:k])}
    group0_ratings = [scores[item_id] for item_id in predicted_list[:k] if group_mapping.get(item_id, 0) == 0]
    group1_ratings = [scores[item_id] for item_id in predicted_list[:k] if group_mapping.get(item_id, 0) == 1]
    if not group0_ratings or not group1_ratings:
        return 0.0
    avg_rating0 = sum(group0_ratings) / len(group0_ratings)
    avg_rating1 = sum(group1_ratings) / len(group1_ratings)
    return abs(avg_rating0 - avg_rating1)


# 异步请求函数（从pasted_text_2.txt中提取）
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


# LLM4Rerank 核心类
class LLM4Rerank:
    def __init__(self, api_key: str, model_name: str = "deepseek-chat", base_url: str = "https://api.deepseek.com"):
        """
        初始化 LLM4Rerank 框架。
        参数:
            api_key (str): API 密钥
            model_name (str): 模型名称，默认为 deepseek-chat（适配您的实际API）
            base_url (str): API 基础URL
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        # 定义节点及其提示模板（基于论文中的模板，适配您的电影推荐任务）
        # 定义节点及其提示模板（基于论文中的模板，适配您的验证集数据）
        self.nodes = {
            "Accuracy": {
                "type": "aspect",
                "system_prompt": "You are a recommendation expert focused on accuracy.",
                "user_prompt_template": (
                    "Considering a user with the following viewing history:\n"
                    "{user_history}\n\n"
                    "Here’s a list of the candidate movies:\n"
                    "{candidate_list}\n\n"
                    "Your reranking goal: {goal}\n"
                    "Your historical reranking: {historical_pool}\n"
                    "Now, you need to focus on the accuracy aspect (the match between the user’s preferences and the movies) "
                    "and rerank the candidates based on the given information, prioritizing movies most likely to be watched next by the user, "
                    "and then give suggestions about the next step of reranking from the following reranking nodes considering the goal: "
                    "{available_nodes}\n"
                    "For your response format: Provide the reranked list of item IDs followed by the next node name, "
                    "e.g., 'Reranked List: [ID1, ID2, ...]; Next Node: Diversity'"
                )
            },
            "Diversity": {
                "type": "aspect",
                "system_prompt": "You are a recommendation expert focused on diversity.",
                "user_prompt_template": (
                    "Considering a user with the following viewing history:\n"
                    "{user_history}\n\n"
                    "Here’s a list of the candidate movies:\n"
                    "{candidate_list}\n\n"
                    "Your reranking goal: {goal}\n"
                    "Your historical reranking: {historical_pool}\n"
                    "Now, you need to focus on the diversity aspect (more movies with different genres or themes should exist at the top of the reranking list) "
                    "and rerank the candidates based on the given information to ensure a varied selection, "
                    "and then give suggestions about the next step of reranking from the following reranking nodes considering the goal: "
                    "{available_nodes}\n"
                    "For your response format: Provide the reranked list of item IDs followed by the next node name, "
                    "e.g., 'Reranked List: [ID1, ID2, ...]; Next Node: Fairness'"
                )
            },
            "Fairness": {
                "type": "aspect",
                "system_prompt": "You are a recommendation expert focused on fairness.",
                "user_prompt_template": (
                    "Considering a user with the following viewing history:\n"
                    "{user_history}\n\n"
                    "Here’s a list of the candidate movies:\n"
                    "{candidate_list}\n\n"
                    "Your reranking goal: {goal}\n"
                    "Your historical reranking: {historical_pool}\n"
                    "Now, you need to focus on the fairness aspect (for movies from different time periods or categories, you should keep the average ranking of the groups similar) "
                    "and rerank the candidates based on the given information to balance exposure across different movie groups, "
                    "and then give suggestions about the next step of reranking from the following reranking nodes considering the goal: "
                    "{available_nodes}\n"
                    "For your response format: Provide the reranked list of item IDs followed by the next node name, "
                    "e.g., 'Reranked List: [ID1, ID2, ...]; Next Node: Stop'"
                )
            },
            "Backward": {
                "type": "functional",
                "system_prompt": "You are a recommendation expert managing the reranking process.",
                "user_prompt_template": (
                    "Considering a user with the following viewing history:\n"
                    "{user_history}\n\n"
                    "Here’s a list of the candidate movies:\n"
                    "{candidate_list}\n\n"
                    "Your reranking goal: {goal}\n"
                    "Your historical reranking: {historical_pool}\n"
                    "Now, you need to give suggestions about the next step of reranking from the following reranking nodes considering the goal: "
                    "{available_nodes}\n"
                    "Note: This is a backward step, so the latest reranking result will be ignored.\n"
                    "For your response format: Provide the next node name, e.g., 'Next Node: Accuracy'"
                )
            },
            "Stop": {
                "type": "functional",
                "system_prompt": "",
                "user_prompt_template": ""  # Stop节点不需要提示模板
            }
        }

        # 可用节点列表
        self.available_nodes = list(self.nodes.keys())
        # 评估指标
        self.metrics = {
            "HR": calculate_hr,
            "NDCG": calculate_ndcg,
            "Alpha_NDCG": calculate_alpha_ndcg,
            "MAD": calculate_mad
        }
        logging.info(f"LLM4Rerank 初始化完成，使用模型: {model_name}")

    def add_node(self, node_name: str, node_type: str, system_prompt: str, user_prompt_template: str) -> None:
        """
        添加新节点到LLM4Rerank框架中。
        参数:
            node_name (str): 节点名称
            node_type (str): 节点类型（aspect 或 functional）
            system_prompt (str): 系统提示词
            user_prompt_template (str): 用户提示词模板
        """
        if node_name in self.nodes:
            logging.warning(f"节点 {node_name} 已存在，将被覆盖")
        self.nodes[node_name] = {
            "type": node_type,
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template
        }
        self.available_nodes = list(self.nodes.keys())
        logging.info(f"添加新节点: {node_name}，类型: {node_type}")

    def construct_prompt(self, data: Dict[str, Any], current_node: str, goal: str,
                         historical_pool: List[Tuple[str, List[int]]]) -> List[Dict[str, str]]:
        if current_node not in self.nodes:
            raise ValueError(f"节点 {current_node} 未定义")
        node_info = self.nodes[current_node]
        if node_info["type"] == "functional" and current_node == "Stop":
            return []  # Stop节点不需要提示

        # 格式化用户历史观影记录（取最近10部电影，与您的代码一致）
        user_history = "\n".join([f"- {movie[1]} (ID: {movie[0]})" for movie in data.get('item_list', [])[-10:]])
        # 格式化候选列表
        candidate_list = "\n".join([f"- {movie[1]} (ID: {movie[0]})" for movie in data.get('candidates', [])])
        # 格式化历史重排池
        historical_str = "None" if not historical_pool else "\n".join(
            [f"{node}: {result}" for node, result in historical_pool]
        )
        # 可用节点列表
        available_nodes_str = ", ".join(self.available_nodes)

        # 构造提示词
        user_prompt = node_info["user_prompt_template"].format(
            user_history=user_history,
            candidate_list=candidate_list,
            goal=goal,
            historical_pool=historical_str,
            available_nodes=available_nodes_str
        )
        return [
            {"role": "system", "content": node_info["system_prompt"]},
            {"role": "user", "content": user_prompt}
        ]

    async def predict(self, prompt_messages: List[Dict[str, str]]) -> Tuple[str, List[int]]:
        """
        使用 LLM 进行预测，获取重排结果和下一个节点。
        参数:
            prompt_messages (List[Dict[str, str]]): 提示词消息列表
        返回:
            Tuple[str, List[int]]: 下一个节点名称和重排结果
        """
        if not prompt_messages:  # Stop节点或其他无需预测的情况
            return "Stop", []
        try:
            output_text = await query_deepseek_async(self.client, prompt_messages)
            # 解析输出
            next_node, reranked_list = self.parse_output(output_text)
            return next_node, reranked_list
        except Exception as e:
            logging.error(f"LLM 预测失败: {e}")
            return "Stop", []

    def parse_output(self, text: str) -> Tuple[str, List[int]]:
        """
        解析 LLM 输出，提取推荐列表和下一个节点，增强容错性。
        参数:
            text (str): LLM 输出文本
        返回:
            Tuple[str, List[int]]: 下一个节点名称和重排列表
        """
        next_node = "Stop"  # 默认终止
        reranked_list = []
        try:
            # 查找重排列表
            list_match = re.search(r'Reranked List: \[([^\]]+)\]', text)
            if list_match:
                list_str = list_match.group(1)
                reranked_list = list(map(int, re.findall(r'\d+', list_str)))
            else:
                # 备选解析方式，兼容其他格式
                ids = re.findall(r'\b\d+\b', text.replace('，', ',').replace(' ', ','))
                seen = set()
                for id_str in ids:
                    if id_str not in seen:
                        seen.add(id_str)
                        reranked_list.append(int(id_str))
            # 查找下一个节点
            node_match = re.search(r'Next Node: (\w+)', text)
            if node_match:
                next_node = node_match.group(1)
                if next_node not in self.available_nodes:
                    next_node = "Stop"
        except Exception as e:
            logging.warning(f"输出解析失败: {e}, 使用默认值")
        logging.info(f"模型输出解析 - 原始输出:\n{text}")
        logging.info(f"模型输出解析 - 解析后电影ID列表: {reranked_list}, 下一个节点: {next_node}")
        return next_node, reranked_list

    async def run_reranking(self, data: Dict[str, Any], goal: str, max_node_count: int = 5) -> List[int]:
        """
        运行自动重排过程（基于论文中的Algorithm 1）。
        参数:
            data (Dict[str, Any]): 数据样本
            goal (str): 重排目标
            max_node_count (int): 最大节点计数，超过则终止
        返回:
            List[int]: 最终重排结果
        """
        current_node = "Accuracy"  # 论文中规定从Accuracy节点开始
        node_count = 0
        historical_pool = []  # 历史重排池，存储 (节点名, 重排结果) 对

        logging.info(f"开始重排过程，目标: {goal}")
        while current_node != "Stop":
            logging.info(f"当前节点: {current_node}, 节点计数: {node_count + 1}")
            prompt = self.construct_prompt(data, current_node, goal, historical_pool)
            next_node, reranked_list = await self.predict(prompt)
            if current_node != "Backward" and reranked_list:
                historical_pool.append((current_node, reranked_list))
            elif current_node == "Backward" and historical_pool:
                # Backward节点移除最新的重排结果
                historical_pool.pop()
                logging.info("执行Backward操作，移除最新重排结果")

            node_count += 1
            if node_count >= max_node_count:
                logging.info(f"达到最大节点计数 {max_node_count}，终止重排")
                current_node = "Stop"
            else:
                current_node = next_node
                logging.info(f"下一个节点: {current_node}")

        # 返回历史重排池中的最新结果
        final_result = historical_pool[-1][1] if historical_pool else []
        logging.info(f"重排完成，最终结果: {final_result}")
        return final_result

    def evaluate(self, predicted_list: List[int], data: Dict[str, Any], k: int = 10, alpha: float = 0.5) -> Dict[
        str, float]:
        """
        评估推荐结果，适配实际数据集。
        参数:
            predicted_list (List[int]): 预测的推荐列表
            data (Dict[str, Any]): 数据样本
            k (int): 前k项
            alpha (float): α-NDCG 的参数
        返回:
            Dict[str, float]: 各指标的值
        """
        target_item = data.get("target_item", [0])[0]
        # 多样性分组（假设数据中包含类别信息，需根据实际字段调整）
        item_groups = {item[0]: "unknown" for item in data.get("candidates", [])}
        # 公平性分组（假设使用年份，需根据实际字段调整）
        group_mapping = {item[0]: 0 for item in data.get("candidates", [])}

        results = {
            "HR": self.metrics["HR"](predicted_list, target_item, k),
            "NDCG": self.metrics["NDCG"](predicted_list, target_item, k),
            "Alpha_NDCG": self.metrics["Alpha_NDCG"](predicted_list, item_groups, k, alpha),
            "MAD": self.metrics["MAD"](predicted_list, group_mapping, k)
        }
        return results


# 结合evaluate_detailed的评估函数
async def evaluate_detailed_llm4rerank(
        val_data: List[Dict[str, Any]],
        api_key: str,
        goal: str = "Mainly focus on accuracy, followed by diversity",
        num_trials: int = 10,
        max_concurrency: int = 10,
        max_node_count: int = 5,
        k: int = 10,
        result_dir: str = "results_llm4rerank"
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """
    详细评估函数，结合LLM4Rerank和evaluate_detailed逻辑，对每个样本进行多次评估。
    参数:
        val_data (List[Dict[str, Any]]): 验证数据集
        api_key (str): API密钥
        goal (str): 重排目标
        num_trials (int): 每个样本的评估次数
        max_concurrency (int): 最大并发数
        max_node_count (int): 每个重排过程的最大节点数
        k (int): 计算NDCG时的前k项
        result_dir (str): 结果保存目录
    返回:
        Tuple[Dict[str, Any], Dict[str, List[float]]]: 详细结果字典和每次评估的平均指标列表
    """
    logging.info(f"\n===== 开始评估 LLM4Rerank，目标: {goal} ======")
    os.makedirs(result_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrency)
    reranker = LLM4Rerank(api_key=api_key, model_name="deepseek-chat", base_url="https://api.deepseek.com")

    num_samples = len(val_data)
    sample_scores = {i: {"HR": [], "NDCG": [], "Alpha_NDCG": [], "MAD": []} for i in range(num_samples)}

    # 对每个样本进行num_trials次评估
    for trial in range(num_trials):
        logging.info(f"开始第 {trial + 1}/{num_trials} 次评估")

        async def worker(sample_idx: int, sample: Dict[str, Any]) -> Dict[str, float]:
            async with semaphore:
                try:
                    final_result = await reranker.run_reranking(sample, goal, max_node_count)
                    metrics = reranker.evaluate(final_result, sample, k)
                    logging.info(f"样本 {sample_idx} (Trial {trial + 1}): 评估结果 {metrics}")
                    return metrics
                except Exception as e:
                    logging.warning(f"样本 {sample_idx} (Trial {trial + 1}) 出错: {e}")
                    return {"HR": 0.0, "NDCG": 0.0, "Alpha_NDCG": 0.0, "MAD": 0.0}

        # 并发处理所有样本
        tasks = [worker(idx, sample) for idx, sample in enumerate(val_data)]
        trial_results = [await t for t in
                         tqdm(asyncio.as_completed(tasks), total=num_samples, desc=f"Trial {trial + 1}")]

        # 记录每个样本的得分
        for idx, metrics in enumerate(trial_results):
            for metric_name, value in metrics.items():
                sample_scores[idx][metric_name].append(value)

    # 计算统计量
    trial_avg_scores = {"HR": [], "NDCG": [], "Alpha_NDCG": [], "MAD": []}
    for trial in range(num_trials):
        for metric_name in trial_avg_scores.keys():
            scores = [sample_scores[idx][metric_name][trial] for idx in range(num_samples)]
            avg_score = sum(scores) / len(scores)
            trial_avg_scores[metric_name].append(avg_score)
            logging.info(f"Trial {trial + 1} 平均 {metric_name}@{k}: {avg_score:.4f}")

    overall_avg_scores = {
        name: sum(scores) / len(scores) for name, scores in trial_avg_scores.items()
    }
    logging.info(f"总体平均评估结果: {overall_avg_scores}")

    # 计算每个样本的平均得分和标准差
    sample_stats = {}
    for idx in range(num_samples):
        stats = {}
        for metric_name in sample_scores[idx].keys():
            scores = sample_scores[idx][metric_name]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            stats[metric_name] = {"mean": mean_score, "std": std_score, "scores": scores}
        sample_stats[idx] = stats
        logging.info(f"样本 {idx}: 平均评估结果 { {m: s['mean'] for m, s in stats.items()} }")

    # 保存详细结果
    detailed_result = {
        "goal": goal,
        "overall_avg_scores": overall_avg_scores,
        "trial_avg_scores": trial_avg_scores,
        "sample_stats": sample_stats,
        "num_trials": num_trials,
        "num_samples": num_samples,
        "k": k,
        "max_node_count": max_node_count
    }

    # 保存为JSON
    result_file = os.path.join(result_dir, f"eval_llm4rerank_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_result, f, indent=2, ensure_ascii=False)
    logging.info(f"详细结果已保存至: {result_file}")

    # 保存为CSV（便于查看每个样本的得分）
    for metric_name in ["HR", "NDCG", "Alpha_NDCG", "MAD"]:
        csv_file = os.path.join(result_dir, f"sample_scores_{metric_name}.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ["Sample_ID", f"Mean_{metric_name}", f"Std_{metric_name}"] + [f"Trial_{i + 1}" for i in
                                                                                   range(num_trials)]
            writer.writerow(header)
            for idx in range(num_samples):
                stats = sample_stats[idx][metric_name]
                row = [idx, f"{stats['mean']:.4f}", f"{stats['std']:.4f}"] + \
                      [f"{score:.4f}" for score in stats['scores']]
                writer.writerow(row)
        logging.info(f"样本得分CSV已保存至: {csv_file}")

    return detailed_result, trial_avg_scores


# 示例主函数
if __name__ == "__main__":
    setup_logging(log_dir="logs_eval_llm4rerank", log_name="detailed_eval.log")

    API_KEY = "sk-60981d4439e34af29f9b9afef32d6c7e"
    VAL_PATH = "E:/PE_Exam/val.jsonl"

    # 加载验证数据
    logging.info(f"加载验证数据 - 路径: {VAL_PATH}")
    try:
        with open(VAL_PATH, 'r', encoding='utf-8') as f:
            val_data = [json.loads(line.strip()) for line in f]
        logging.info(f"验证数据加载成功 - 样本数: {len(val_data)}")
    except Exception as e:
        logging.error(f"验证数据加载失败: {e}")
        val_data = []

    if val_data:
        # 运行详细评估
        results, trial_scores = asyncio.run(evaluate_detailed_llm4rerank(
            val_data=val_data,
            api_key=API_KEY,
            goal="Mainly focus on accuracy, followed by diversity",
            num_trials=10,
            max_concurrency=10,
            max_node_count=5,
            k=10,
            result_dir="results_llm4rerank"
        ))
        logging.info("评估完成")
        print("总体平均评估结果:", results["overall_avg_scores"])
    else:
        logging.warning("无验证数据，跳过评估")
