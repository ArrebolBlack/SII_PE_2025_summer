import json
import asyncio
import math
import logging
import os
import re
import csv
from datetime import datetime
from tqdm import tqdm
from openai import AsyncOpenAI


# 设置日志配置
def setup_logging(log_dir="logs_ape"):
    """
    设置日志记录配置，将日志保存到文件。
    参数:
        log_dir (str): 日志文件存储目录
    返回:
        None
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ape_optimization_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logging.info("APE 日志系统初始化完成")


async def optimize_prompt_strategy(client, iteration, history_results):
    """
    使用思考规划模型优化提示词策略，直接生成提示词模板
    参数:
        client: AsyncOpenAI 客户端
        iteration: 当前迭代轮次
        history_results: 历史评估结果列表 [{'strategy': {...}, 'score': float}, ...]
    返回:
        dict: 新的提示词策略，包含直接生成的系统提示词和用户提示词模板
    """
    # 构建优化提示词
    system_prompt = "你是一名提示词工程专家，专注于设计有效的提示词以提升大语言模型在电影推荐重排任务中的表现。你的目标是根据用户历史观影记录和候选电影列表，设计一个提示词模板，使模型能够准确预测用户下一步可能观看的电影。"

    history_summary = "用户历史观影记录格式：用户最近观看的电影列表（按时间顺序，越靠后越近期）。\n"
    candidates_summary = "候选电影列表格式：一组电影ID和名称的列表，包含用户实际下一步观看的电影。\n"
    task_summary = "任务：对候选电影进行重排，使最可能被用户观看的电影排在最前面，并输出电影ID列表。\n"
    eval_metric = "评估指标：NDCG@10，分数范围[0,1]，越高越好。\n"

    # 结构化历史优化轨迹
    history_text = "\n".join([
        f"迭代 {i + 1}: 策略名称: {res['strategy']['name']}, NDCG@10: {res['score']:.4f}\n"
        f"系统提示词: {res['strategy'].get('system_prompt', '无')}\n"
        f"用户提示词模板: {res['strategy'].get('user_prompt_template', '无')}\n"
        for i, res in enumerate(history_results)
    ]) if history_results else "暂无历史优化记录。\n"

    user_prompt = f"当前迭代轮次：{iteration}\n\n任务背景：\n{history_summary}{candidates_summary}{task_summary}{eval_metric}\n\n" \
                  f"历史优化轨迹：\n{history_text}\n\n" \
                  f"请分析历史策略的优缺点，提出改进建议，并设计一个新的提示词模板。输出格式为JSON：\n" \
                  f"{{\n  'name': '策略名称',\n  'system_prompt': '完整的系统提示词内容',\n  'user_prompt_template': '用户提示词模板，包含{{history}}和{{candidates}}占位符',\n  'analysis': '历史策略分析',\n  'improvement': '改进建议'\n}}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",  # 使用强大的模型作为思考规划模型
            messages=messages,
            temperature=0.7,  # 允许一定的创造性
            max_tokens=1024
        )
        output_text = response.choices[0].message.content
        logging.info(f"优化提示词策略 - 模型输出:\n{output_text}")

        # 解析模型输出的JSON
        start_idx = output_text.find("{")
        end_idx = output_text.rfind("}") + 1
        json_str = output_text[start_idx:end_idx]
        new_strategy = json.loads(json_str)
        logging.info(f"优化提示词策略 - 新策略: {new_strategy}")
        return new_strategy
    except Exception as e:
        logging.error(f"优化提示词策略失败: {e}")
        # 如果失败，返回一个默认策略
        return {
            "name": f"iteration_{iteration}_fallback",
            "system_prompt": "你是一名电影推荐专家，目标是根据用户历史预测他们最可能观看的电影。",
            "user_prompt_template": "用户最近观看的电影：\n{{history}}\n\n请根据用户的兴趣，对以下候选电影进行排序（最可能观看的电影在最前）：\n{{candidates}}\n\n直接输出电影ID列表，不要额外的解释或文字。"
        }


def construct_prompt_direct(d, strategy):
    """根据模型直接生成的提示词模板构建提示词"""
    system_prompt = strategy['system_prompt']
    user_prompt_template = strategy['user_prompt_template']

    # 格式化历史记录和候选列表
    history_text = "\n".join([f"- {movie[1]}" for movie in d['item_list'][-10:]])
    candidates_text = "\n".join([f"{movie[0]}: {movie[1]}" for movie in d['candidates']])

    # 替换模板中的占位符
    user_prompt = user_prompt_template.replace("{{history}}", history_text).replace("{{candidates}}", candidates_text)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


async def evaluate_strategy(val_data, api_key, strategy, max_concurrency=10, num_repeats=10):
    """
    异步评估提示词策略的性能，重复多次取平均值
    参数:
        val_data: 验证数据集
        api_key: DeepSeek API 密钥
        strategy: 提示词策略字典
        max_concurrency: 最大并发请求数
        num_repeats: 重复评估次数，以处理得分不稳定性
    返回:
        tuple: (策略名称, 平均NDCG分数)
    """
    logging.info(f"\n===== 评估策略: {strategy['name']}，重复次数: {num_repeats} =====")
    all_scores = []

    for repeat in range(num_repeats):
        logging.info(f"重复评估 {repeat + 1}/{num_repeats}")
        semaphore = asyncio.Semaphore(max_concurrency)
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        async def worker(d, idx):
            async with semaphore:
                prompt = construct_prompt_direct(d, strategy)
                try:
                    output_text = await client.chat.completions.create(
                        model="deepseek-chat",  # 使用DeepSeek-V3或其他指定模型
                        messages=prompt,
                        temperature=0,  # 确保确定性输出
                        max_tokens=100
                    )
                    output_text = output_text.choices[0].message.content
                    predicted_list = list(map(int, re.findall(r'\d+', output_text)))
                    ndcg = calculate_ndcg_for_sample(predicted_list, d["target_item"][0])
                    return ndcg
                except Exception as e:
                    logging.warning(f"出错样本 (索引: {idx}): {e}")
                    return 0.0

        tasks = [worker(d, idx) for idx, d in enumerate(val_data)]
        scores = [await t for t in
                  tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"评估 {repeat + 1}/{num_repeats}")]
        avg_ndcg_repeat = sum(scores) / len(scores) if scores else 0.0
        logging.info(f"重复 {repeat + 1}/{num_repeats} - 平均 NDCG@10: {avg_ndcg_repeat:.4f}")
        all_scores.append(avg_ndcg_repeat)

    avg_ndcg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    logging.info(f"策略 {strategy['name']} 最终平均 NDCG@10 (共 {num_repeats} 次): {avg_ndcg:.4f}\n")
    return strategy['name'], avg_ndcg


def calculate_ndcg_for_sample(predicted_list, ground_truth_item, k=10):
    predicted_list = predicted_list[:k]
    relevance = [1 if item_id == ground_truth_item else 0 for item_id in predicted_list]

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
    idcg = 1.0  # 只有一个相关项置于首位的情况

    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg


async def run_ape_optimization(val_path, api_key, max_iterations=5, val_data_limit=50, num_repeats=10):
    """
    运行自动提示工程师(APE)优化过程
    参数:
        val_path: 验证数据路径
        api_key: DeepSeek API 密钥
        max_iterations: 最大迭代轮次
        val_data_limit: 限制验证数据样本数以加速评估，None 表示使用全部数据
        num_repeats: 重复评估次数，以处理得分不稳定性
    """
    # 加载验证数据
    try:
        with open(val_path, 'r', encoding='utf-8') as f:
            val_data = [json.loads(line.strip()) for line in f if line.strip()]
        if not val_data:
            logging.error("验证数据为空，无法进行评估")
            return
        logging.info(f"加载验证数据成功，样本数: {len(val_data)}")
        # 限制验证数据样本数以加速评估
        if val_data_limit is not None:
            val_data = val_data[:val_data_limit]
            logging.info(f"限制验证数据样本数为 {val_data_limit} 以加速评估")
        else:
            logging.info("使用完整验证数据集进行评估")
    except Exception as e:
        logging.error(f"加载验证数据失败: {e}")
        return

    # 初始化思考规划模型客户端
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # 初始策略
    current_strategy = {
        "name": "initial_expert_strategy",
        "system_prompt": "你是一名电影推荐专家，目标是根据用户历史预测他们最可能观看的电影。",
        "user_prompt_template": "用户最近观看的电影：\n{{history}}\n\n请根据用户的兴趣，对以下候选电影进行排序（最可能观看的电影在最前）：\n{{candidates}}\n\n直接输出电影ID列表，不要额外的解释或文字。"
    }
    history_results = []

    # 多轮迭代优化
    for iteration in range(1, max_iterations + 1):
        logging.info(f"\n======== APE 优化迭代轮次 {iteration}/{max_iterations} =========")

        # 评估当前策略
        name, score = await evaluate_strategy(val_data, api_key, current_strategy, num_repeats=num_repeats)
        history_results.append({"strategy": current_strategy, "score": score})

        # 使用思考规划模型优化策略
        new_strategy = await optimize_prompt_strategy(client, iteration, history_results)
        current_strategy = new_strategy

    # 最终结果汇总
    logging.info("\n======== APE 优化完成，最终结果 =========")
    best_result = max(history_results, key=lambda x: x['score'], default={"strategy": {"name": "无策略"}, "score": 0.0})
    logging.info(f"最佳策略: {best_result['strategy']['name']}, NDCG@10: {best_result['score']:.4f}")
    print(f"最佳策略: {best_result['strategy']['name']}, NDCG@10: {best_result['score']:.4f}")

    # 保存最终策略和结果
    output_dir = "result/ape"
    os.makedirs(output_dir, exist_ok=True)

    # 保存最终策略
    with open(os.path.join(output_dir, "final_strategy.json"), "w", encoding="utf-8") as f:
        json.dump(current_strategy, f, ensure_ascii=False, indent=2)
    logging.info("最终策略已保存到 final_strategy.json")

    # 保存最佳策略
    with open(os.path.join(output_dir, "best_strategy.json"), "w", encoding="utf-8") as f:
        json.dump(best_result['strategy'], f, ensure_ascii=False, indent=2)
    logging.info("最佳策略已保存到 best_strategy.json")

    # 保存评估结果
    with open(os.path.join(output_dir, "evaluation_results.csv"), "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["迭代轮次", "策略名称", "NDCG@10"])
        for i, res in enumerate(history_results):
            writer.writerow([i + 1, res['strategy']['name'], f"{res['score']:.4f}"])
    logging.info("评估结果已保存到 evaluation_results.csv")

    # 保存策略历史记录
    with open(os.path.join(output_dir, "strategies_history.json"), "w", encoding="utf-8") as f:
        json.dump(history_results, f, ensure_ascii=False, indent=2)
    logging.info("策略历史记录已保存到 strategies_history.json")


if __name__ == "__main__":
    # 初始化日志系统
    setup_logging()

    API_KEY = "sk-60981d4439e34af29f9b9afef32d6c7e"  # 建议使用环境变量
    VAL_PATH = "E:/PE_Exam/val.jsonl"
    logging.info("程序启动 - DeepSeek API 密钥已加载（已隐藏）")

    # 运行APE优化过程
    asyncio.run(run_ape_optimization(VAL_PATH, API_KEY, max_iterations=30, val_data_limit=20, num_repeats=10))
