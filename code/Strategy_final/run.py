import json
import asyncio
import math
import logging
from tqdm import tqdm
from openai import AsyncOpenAI
import re
import csv
import os
from datetime import datetime, timedelta
import itertools
import time
import argparse

# 计算 NDCG@10
def calculate_ndcg_for_sample(predicted_list, ground_truth_item, k=10):
    relevance = [1 if item_id == ground_truth_item else 0 for item_id in predicted_list[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
    idcg = 1 / math.log2(1 + 1)  # 理想情况：相关项在第一位
    ndcg = dcg / idcg if idcg > 0 else 0
    logging.info(f"NDCG计算 - 预测列表前{k}项: {predicted_list[:k]}, 真实目标: {ground_truth_item}, NDCG@{k}: {ndcg}")
    return ndcg

# 解析模型输出
def parse_output(text, ranker_type="Listwise"):
    """
    解析大语言模型的输出文本，提取推荐重排列表或评分
    参数:
        text (str): 大语言模型在设计prompt下的输出文本
        ranker_type (str): 排名方法类型，用于调整解析逻辑
    返回:
        list or float: 根据排名方法返回电影ID列表或单个评分
    """
    logging.info(f"模型输出解析 - 排名方法: {ranker_type}, 原始输出:\n{text}")
    if ranker_type == "Pointwise":
        # 提取评分或相关性判断
        text_lower = text.lower()
        if "yes" in text_lower or "highly relevant" in text_lower:
            return 1.0
        elif "somewhat relevant" in text_lower:
            return 0.5
        elif "no" in text_lower or "not relevant" in text_lower:
            return 0.0
        # 尝试提取数值评分
        matches = re.findall(r'\d+(\.\d+)?', text)
        if matches:
            try:
                return float(matches[0])
            except:
                pass
        return 0.0  # 默认评分
    elif ranker_type == "Pairwise":
        # 提取比较结果，假设输出为 "Movie A" 或 "Movie B"
        text_lower = text.lower()
        if "movie a" in text_lower or "passage a" in text_lower:
            return "A"
        elif "movie b" in text_lower or "passage b" in text_lower:
            return "B"
        return "A"  # 默认值
    else:  # Listwise 和 Setwise
        # 提取数字ID列表
        parsed_list = list(map(int, re.findall(r'\d+', text)))
        if not parsed_list:
            logging.warning("解析失败，未找到数字ID列表")
            return []
        logging.info(f"模型输出解析 - 解析后电影ID列表: {parsed_list}")
        return parsed_list

# 异步请求函数（从示例代码中提取，不包含重试逻辑）
async def query_deepseek_async(client, prompt_messages):
    """
    异步请求 DeepSeek API，无重试逻辑
    参数:
        client: AsyncOpenAI 客户端
        prompt_messages: 提示消息列表
    返回:
        str: 模型输出文本
    """
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
log_dir = "logs_Strategy"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"evaluation_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()  # 保留控制台输出以便监控
    ]
)

# 定义论文中的提示组件（基于 Table 2）
PROMPT_COMPONENTS = {
    "TaskInstruction": {
        "Pointwise": [
            "Does the movie match the user's viewing history and preferences?",
            "Is this movie relevant to the user's interests based on their history?",
            "For the following user history and movie, judge whether they are relevant.",
            "Judge the relevance between the user's history and the movie."
        ],
        "Pairwise": ["Given a user's viewing history, which of the following two movies is more relevant to their interests?"],
        "Listwise": [
            "Rank the {num} movies based on their relevance to the user's viewing history and preferences.",
            "Sort the movies by their relevance to the user's interests based on history.",
            "I will provide you with {num} movies, each indicated by number identifier []. Rank the movies based on their relevance to the user's history."
        ],
        "Setwise": ["Which movie is the most relevant to the user's viewing history and preferences?"]
    },
    "OutputType": {
        "Pointwise": [
            "Judge whether they are 'Highly Relevant', 'Somewhat Relevant', or 'Not Relevant'.",
            "From a scale of 0 to 4, judge the relevance.",
            "Answer 'Yes' or 'No'.",
            "Answer True/False."
        ],
        "Pairwise": ["Output Movie A or Movie B."],
        "Listwise": ["The movies should be listed in descending order using identifiers. The most relevant movies should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only output the ranking results, do not explain."],
        "Setwise": [
            "Output the movie ID of the most relevant movie.",
            "Generate the movie ID.",
            "Generate the movie ID that is the most relevant to the user's history, then explain why you think this movie is the most relevant."
        ]
    },
    "ToneWords": [
        "",  # 无语气词
        "You better get this right or you will be punished.",
        "Only output the ranking results, do not say any word or explanation.",
        "Please",
        "Only",
        "Must"
    ],
    "RolePlaying": [
        "",  # 无角色扮演
        "You are RankGPT, an intelligent assistant that can rank movies based on their relevancy to the user's viewing history and preferences."
    ]
}

# 构造提示变体函数，融合论文中的策略
def construct_prompt_variant(d, strategy, movie_subset=None, iteration=None):
    """
    根据策略组合生成 Prompt 消息列表，融合论文中的提示变体
    参数:
        d (dict): jsonl 数据文件的一行，解析成字典后的变量
        strategy (dict): 包含排名方法和提示组件组合的策略字典
        movie_subset (list): 用于 Pairwise 和 Setwise 的电影子集，默认为 None 表示使用所有候选电影
        iteration (int): 用于 Setwise 的迭代轮次，默认为 None
    返回:
        list: OpenAI API 的 message 格式列表
    """
    ranker_type = strategy.get("ranker_type", "Listwise")
    ti_idx = strategy.get("ti_idx", 0)
    ot_idx = strategy.get("ot_idx", 0)
    tw_idx = strategy.get("tw_idx", 0)
    rp_idx = strategy.get("rp_idx", 0)
    eo = strategy.get("evidence_ordering", "QF")
    pe = strategy.get("position_evidence", "B")

    # 获取提示组件内容
    ti_list = PROMPT_COMPONENTS["TaskInstruction"][ranker_type]
    ot_list = PROMPT_COMPONENTS["OutputType"][ranker_type]
    tw_list = PROMPT_COMPONENTS["ToneWords"]
    rp_list = PROMPT_COMPONENTS["RolePlaying"]

    task_instruction = ti_list[min(ti_idx, len(ti_list) - 1)]
    output_type = ot_list[min(ot_idx, len(ot_list) - 1)]
    tone_words = tw_list[min(tw_idx, len(tw_list) - 1)]
    role_playing = rp_list[min(rp_idx, len(rp_list) - 1)]

    # 替换 {num} 占位符
    if movie_subset is None:
        movie_subset = d['candidates']
    task_instruction = task_instruction.replace("{num}", str(len(movie_subset)))

    # 格式化用户历史和候选电影
    history_text = "\n".join([f"[{idx+1}] {m[1]}" for idx, m in enumerate(d['item_list'][-10:])])
    candidates_text = "\n".join([f"[{idx+1}] {i[0]}: {i[1]}" for idx, i in enumerate(movie_subset)])

    # 根据 Evidence Ordering 构建证据部分
    if eo == "QF":
        evidence = f"User's Recent Viewing History:\n{history_text}\n\nCandidate Movies to Rank:\n{candidates_text}"
    else:  # PF
        evidence = f"Candidate Movies to Rank:\n{candidates_text}\n\nUser's Recent Viewing History:\n{history_text}"

    # 构建系统提示
    system_prompt = role_playing if role_playing else "You are an expert in movie recommendation, tasked with predicting the next movie a user is most likely to watch based on their history."

    # 构建用户提示
    user_prompt_components = []
    if pe == "B":
        user_prompt_components.append(evidence)
    user_prompt_components.append(task_instruction)
    if tone_words:
        user_prompt_components.append(tone_words)
    user_prompt_components.append(output_type)
    if pe == "E":
        user_prompt_components.append(evidence)

    # 对于 Pairwise，特殊处理两个电影的标识
    if ranker_type == "Pairwise" and movie_subset and len(movie_subset) == 2:
        candidates_text_pairwise = f"Movie A: {movie_subset[0][0]} - {movie_subset[0][1]}\nMovie B: {movie_subset[1][0]} - {movie_subset[1][1]}"
        if eo == "QF":
            evidence = f"User's Recent Viewing History:\n{history_text}\n\nCandidate Movies to Compare:\n{candidates_text_pairwise}"
        else:
            evidence = f"Candidate Movies to Compare:\n{candidates_text_pairwise}\n\nUser's Recent Viewing History:\n{history_text}"
        if pe == "B":
            user_prompt_components[0] = evidence
        else:
            user_prompt_components[-1] = evidence

    # 对于 Setwise，添加迭代信息
    if ranker_type == "Setwise" and iteration is not None:
        user_prompt_components.insert(1, f"Iteration {iteration + 1}: Select the most relevant movie from the remaining candidates.")

    user_prompt = "\n\n".join(user_prompt_components)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# 实现排名逻辑，增加并发限制
async def rank_movies(client, d, strategy, max_concurrency=10):
    """
    根据排名方法生成最终排序列表
    参数:
        client: AsyncOpenAI 客户端
        d (dict): 数据样本
        strategy (dict): 策略字典
        max_concurrency (int): 最大并发请求数量
    返回:
        list: 排序后的电影ID列表
    """
    ranker_type = strategy.get("ranker_type", "Listwise")
    candidates = d['candidates']
    movie_ids = [c[0] for c in candidates]
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_query(prompt):
        async with semaphore:
            return await query_deepseek_async(client, prompt)

    if ranker_type == "Pointwise":
        # 为每个电影单独评分
        scores = []
        tasks = []
        for i, movie in enumerate(candidates):
            prompt = construct_prompt_variant(d, strategy, movie_subset=[movie])
            tasks.append(limited_query(prompt))
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                logging.warning(f"Pointwise 评分失败 for movie {movie_ids[i]}: {resp}")
                scores.append((movie_ids[i], 0.0))
            else:
                score = parse_output(resp, ranker_type="Pointwise")
                scores.append((movie_ids[i], score))
        # 按评分降序排列
        scores.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in scores]

    elif ranker_type == "Pairwise":
        # 成对比较，构建偏序
        comparisons = []
        tasks = []
        pairs = list(itertools.combinations(candidates, 2))
        for pair in pairs:
            prompt = construct_prompt_variant(d, strategy, movie_subset=pair)
            tasks.append(limited_query(prompt))
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                logging.warning(f"Pairwise 比较失败 for pair {pairs[i]}: {resp}")
                winner = "A"  # 默认值
            else:
                winner = parse_output(resp, ranker_type="Pairwise")
            movie_a, movie_b = pairs[i]
            if winner == "A":
                comparisons.append((movie_a[0], movie_b[0]))  # A > B
            else:
                comparisons.append((movie_b[0], movie_a[0]))  # B > A
        # 简单投票排序
        wins = {sid: 0 for sid in movie_ids}
        for winner, loser in comparisons:
            wins[winner] += 1
        ranked = sorted(movie_ids, key=lambda sid: wins[sid], reverse=True)
        return ranked

    elif ranker_type == "Listwise":
        # 一次性生成排序列表
        prompt = construct_prompt_variant(d, strategy)
        output_text = await limited_query(prompt)
        ranked_list = parse_output(output_text, ranker_type="Listwise")
        # 确保所有电影ID都在列表中
        ranked_ids = [id for id in ranked_list if id in movie_ids]
        missing_ids = [id for id in movie_ids if id not in ranked_ids]
        return ranked_ids + missing_ids

    elif ranker_type == "Setwise":
        # 迭代选择最相关电影
        remaining_candidates = candidates.copy()
        ranked = []
        for iteration in range(len(candidates)):
            prompt = construct_prompt_variant(d, strategy, movie_subset=remaining_candidates, iteration=iteration)
            output_text = await limited_query(prompt)
            selected_list = parse_output(output_text, ranker_type="Setwise")
            if selected_list and selected_list[0] in [c[0] for c in remaining_candidates]:
                selected_id = selected_list[0]
                ranked.append(selected_id)
                remaining_candidates = [c for c in remaining_candidates if c[0] != selected_id]
            else:
                # 如果未找到有效选择，停止迭代
                break
        # 添加剩余电影
        remaining_ids = [c[0] for c in remaining_candidates]
        return ranked + remaining_ids

    return movie_ids  # 默认返回未排序列表

# 异步评测函数，增加并发限制
async def evaluate_strategy(val_data, api_key, strategy, num_repeats=3, max_concurrency=10):
    """
    评估指定策略，返回多次评估的平均得分
    参数:
        val_data: 验证数据集
        api_key: API 密钥
        strategy: 策略字典
        num_repeats: 重复评估次数
        max_concurrency (int): 最大并发请求数量
    返回:
        tuple: (策略名称, 平均 NDCG@10 分数, 每次评估的分数列表)
    """
    logging.info(f"\n===== 评估策略: {strategy['name']} (重复 {num_repeats} 次, 最大并发数: {max_concurrency}) =====")
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def worker(d):
        async with semaphore:
            try:
                ranked_list = await rank_movies(client, d, strategy, max_concurrency=max_concurrency)
                if not ranked_list:
                    return 0.0
                return calculate_ndcg_for_sample(ranked_list, d["target_item"][0])
            except Exception as e:
                logging.warning(f"出错样本: {e}")
                return 0.0

    all_scores = []
    for repeat in range(num_repeats):
        tasks = [worker(d) for d in val_data]
        scores = [await t for t in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                                        desc=f"评估 {strategy['name']} (第 {repeat + 1}/{num_repeats} 次)")]
        avg_ndcg = sum(scores) / len(scores) if scores else 0.0
        all_scores.append(avg_ndcg)
        logging.info(f"策略 {strategy['name']} 第 {repeat + 1}/{num_repeats} 次评估，平均 NDCG@10: {avg_ndcg:.4f}")

    final_avg_ndcg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    logging.info(f"策略 {strategy['name']} 最终平均 NDCG@10 (基于 {num_repeats} 次评估): {final_avg_ndcg:.4f}\n")
    return strategy['name'], final_avg_ndcg, all_scores

# 主运行函数，尝试所有策略组合
async def run_all_strategies(val_path, api_key, max_concurrency=10):
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line.strip()) for line in f]

    # 生成所有策略组合，基于论文中的提示变体
    strategies = []
    ranker_types = ["Pointwise", "Pairwise", "Listwise", "Setwise"]
    evidence_orderings = ["QF", "PF"]  # Query First, Passage First
    position_evidences = ["B", "E"]    # Beginning, End

    for ranker in ranker_types:
        ti_options = range(len(PROMPT_COMPONENTS["TaskInstruction"][ranker]))
        ot_options = range(len(PROMPT_COMPONENTS["OutputType"][ranker]))
        tw_options = range(len(PROMPT_COMPONENTS["ToneWords"]))
        rp_options = range(len(PROMPT_COMPONENTS["RolePlaying"]))
        for ti_idx in ti_options:
            for ot_idx in ot_options:
                for tw_idx in tw_options:
                    for rp_idx in rp_options:
                        for eo in evidence_orderings:
                            for pe in position_evidences:
                                strategy_name = f"{ranker}_TI{ti_idx}_OT{ot_idx}_TW{tw_idx}_RP{rp_idx}_EO{eo}_PE{pe}"
                                strategies.append({
                                    "name": strategy_name,
                                    "ranker_type": ranker,
                                    "ti_idx": ti_idx,
                                    "ot_idx": ot_idx,
                                    "tw_idx": tw_idx,
                                    "rp_idx": rp_idx,
                                    "evidence_ordering": eo,
                                    "position_evidence": pe
                                })

    total_strategies = len(strategies)
    logging.info(f"总共生成 {total_strategies} 个策略组合")

    # 由于策略数量较多，测试时可限制数量
    # strategies = strategies[:10]  # 测试时可取消注释以减少计算量

    results = []
    num_repeats = 3  # 重复评估次数
    start_time = time.time()
    strategy_times = []  # 记录每个策略的执行时间

    print(f"\n开始评估 {total_strategies} 个策略组合...")
    logging.info(f"开始评估 {total_strategies} 个策略组合...")

    # 使用 tqdm 创建整体进度条
    with tqdm(total=total_strategies, desc="整体程序进度", unit="strategy") as pbar:
        for i, strat in enumerate(strategies):
            strat_start_time = time.time()
            name, avg_score, individual_scores = await evaluate_strategy(val_data, api_key, strat, num_repeats, max_concurrency)
            strat_end_time = time.time()
            strat_duration = strat_end_time - strat_start_time
            strategy_times.append(strat_duration)
            results.append((name, avg_score, individual_scores))

            # 更新进度条
            pbar.update(1)

            # 计算进度百分比
            progress_percent = ((i + 1) / total_strategies) * 100

            # 估算剩余时间和预计完成时间
            avg_time_per_strategy = sum(strategy_times) / len(strategy_times) if strategy_times else 0
            remaining_strategies = total_strategies - (i + 1)
            estimated_remaining_time = avg_time_per_strategy * remaining_strategies
            remaining_time_str = str(timedelta(seconds=int(estimated_remaining_time)))
            estimated_completion_time = datetime.now() + timedelta(seconds=int(estimated_remaining_time))
            completion_time_str = estimated_completion_time.strftime("%Y-%m-%d %H:%M:%S")

            # 更新进度条后缀
            pbar.set_postfix({
                "进度": f"{progress_percent:.1f}% ({i+1}/{total_strategies})",
                "预计剩余时间": remaining_time_str,
                "预计完成时间": completion_time_str
            })

            # 记录进度信息到日志和控制台
            progress_msg = (f"进度: {progress_percent:.1f}% ({i+1}/{total_strategies} 策略), "
                           f"当前策略: {name}, 执行时间: {strat_duration:.2f} 秒, "
                           f"预计剩余时间: {remaining_time_str}, 预计完成时间: {completion_time_str}")
            logging.info(progress_msg)
            print(progress_msg)

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    completion_msg = f"整体程序执行完成，总耗时: {total_time_str}"
    logging.info(completion_msg)
    print(completion_msg)

    # 输出结果并记录重复次数
    print("\n======= 策略评测结果汇总 =======")
    print(f"| 策略名称 | 平均 NDCG@10 (基于 {num_repeats} 次评估) |")
    print("|----------|---------------------------|")
    for name, score, _ in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"| {name} | {score:.4f} |")

    # 保存完整结果
    os.makedirs("result", exist_ok=True)
    with open("result/prompt_strategy_results.md", "w", encoding="utf-8") as f:
        f.write(f"| 策略名称 | 平均 NDCG@10 (基于 {num_repeats} 次评估) |\n")
        f.write("|----------|---------------------------|\n")
        for name, score, _ in sorted(results, key=lambda x: x[1], reverse=True):
            f.write(f"| {name} | {score:.4f} |\n")

    with open("result/prompt_strategy_results.csv", "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["策略名称", f"平均 NDCG@10 (基于 {num_repeats} 次评估)"])
        for name, score, _ in sorted(results, key=lambda x: x[1], reverse=True):
            writer.writerow([name, f"{score:.4f}"])

    # 保存详细结果，包括每次评估的分数
    with open("result/prompt_strategy_results_detailed.csv", "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ["策略名称", f"平均 NDCG@10 (基于 {num_repeats} 次评估)"] + [f"第 {i+1} 次评估" for i in range(num_repeats)]
        writer.writerow(header)
        for name, score, individual_scores in sorted(results, key=lambda x: x[1], reverse=True):
            row = [name, f"{score:.4f}"] + [f"{s:.4f}" for s in individual_scores]
            writer.writerow(row)

    # 筛选高分策略（例如，NDCG@10 >= 阈值）
    threshold = 0.5  # 可根据需要调整阈值
    high_performing_strategies = [(name, score) for name, score, _ in results if score >= threshold]
    logging.info(f"高分策略筛选 (阈值 NDCG@10 >= {threshold}): 共 {len(high_performing_strategies)} 个策略")
    for name, score in high_performing_strategies:
        logging.info(f"高分策略: {name}, 平均 NDCG@10: {score:.4f} (基于 {num_repeats} 次评估)")

    # 保存高分策略到单独文件
    with open("result/high_performing_strategies.md", "w", encoding="utf-8") as f:
        f.write(f"| 策略名称 | 平均 NDCG@10 (基于 {num_repeats} 次评估) |\n")
        f.write("|----------|---------------------------|\n")
        for name, score in sorted(high_performing_strategies, key=lambda x: x[1], reverse=True):
            f.write(f"| {name} | {score:.4f} |\n")

    with open("result/high_performing_strategies.csv", "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["策略名称", f"平均 NDCG@10 (基于 {num_repeats} 次评估)"])
        for name, score in sorted(high_performing_strategies, key=lambda x: x[1], reverse=True):
            writer.writerow([name, f"{score:.4f}"])

    print(f"\n高分策略 (NDCG@10 >= {threshold}) 已保存到 result/high_performing_strategies.csv 和 .md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero-shot LLM rankers with adjustable concurrency.")
    parser.add_argument("--max-concurrency", type=int, default=10, help="Maximum number of concurrent API requests.")
    parser.add_argument("--api-key", type=str, default="sk-7dc37cd06ab34eceb9138868cd871eb9", help="DeepSeek API key.")
    parser.add_argument("--val-path", type=str, default="E:/PE_Exam/val.jsonl", help="Path to validation data file.")
    args = parser.parse_args()

    API_KEY = args.api_key
    VAL_PATH = args.val_path
    MAX_CONCURRENCY = args.max_concurrency
    logging.info(f"设置最大并发数: {MAX_CONCURRENCY}")
    asyncio.run(run_all_strategies(VAL_PATH, API_KEY, MAX_CONCURRENCY))
