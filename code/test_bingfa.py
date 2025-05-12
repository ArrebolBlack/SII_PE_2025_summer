import asyncio
import logging
import time
from datetime import datetime
import argparse
from openai import AsyncOpenAI
from tqdm import tqdm
import os

# 设置日志
log_dir = "logs_Concurrency_Test"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"concurrency_test_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 简单的测试提示消息
TEST_PROMPT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, can you help me with a quick question?"}
]


# 异步请求函数
async def query_deepseek_async(client, prompt_messages):
    """
    异步请求 DeepSeek API
    参数:
        client: AsyncOpenAI 客户端
        prompt_messages: 提示消息列表
    返回:
        str: 模型输出文本或错误信息
    """
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=prompt_messages,
            temperature=0,
            max_tokens=100  # 限制输出长度，加快测试速度
        )
        output_text = response.choices[0].message.content
        return output_text
    except Exception as e:
        logging.error(f"API 请求失败: {e}")
        return str(e)


# 测试并发请求
async def test_concurrency(client, concurrency_level):
    """
    测试指定并发级别的API请求
    参数:
        client: AsyncOpenAI 客户端
        concurrency_level: 并发请求数量
    返回:
        tuple: (成功请求数, 失败请求数)
    """
    logging.info(f"测试并发级别: {concurrency_level}")
    tasks = [query_deepseek_async(client, TEST_PROMPT) for _ in range(concurrency_level)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = sum(1 for result in results if not isinstance(result, Exception) and not result.startswith("Error"))
    failure_count = concurrency_level - success_count

    logging.info(f"并发级别 {concurrency_level} 测试结果 - 成功: {success_count}, 失败: {failure_count}")
    return success_count, failure_count


# 主函数：逐步增加并发数直到失败或达到上限
async def find_max_concurrency(api_key, start_concurrency=1, max_concurrency_limit=100, step=1, failure_threshold=0.2):
    """
    逐步增加并发请求数，寻找API的最大并发承受能力
    参数:
        api_key: API 密钥
        start_concurrency: 初始并发数
        max_concurrency_limit: 最大并发数测试上限
        step: 每次增加的并发数
        failure_threshold: 失败率阈值，超过此阈值认为并发数超出API能力
    返回:
        int: 估计的最大并发数
    """
    logging.info("开始测试API最大并发数...")
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    current_concurrency = start_concurrency
    max_successful_concurrency = start_concurrency

    with tqdm(total=max_concurrency_limit, desc="并发测试进度", unit="level") as pbar:
        while current_concurrency <= max_concurrency_limit:
            pbar.update(step)
            success_count, failure_count = await test_concurrency(client, current_concurrency)
            failure_rate = failure_count / current_concurrency if current_concurrency > 0 else 0

            if failure_rate > failure_threshold:
                logging.warning(
                    f"并发级别 {current_concurrency} 失败率 {failure_rate:.2%} 超过阈值 {failure_threshold:.2%}，停止测试")
                break
            else:
                max_successful_concurrency = current_concurrency
                logging.info(f"并发级别 {current_concurrency} 测试通过，失败率 {failure_rate:.2%}")

            current_concurrency += step
            # 可选：添加小幅延迟以避免过于频繁的请求
            await asyncio.sleep(1)

    logging.info(f"API最大并发数估计为: {max_successful_concurrency}")
    return max_successful_concurrency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the maximum concurrency level for DeepSeek API.")
    parser.add_argument("--api-key", type=str, default="sk-7dc37cd06ab34eceb9138868cd871eb9", help="DeepSeek API key.")
    parser.add_argument("--start-concurrency", type=int, default=30, help="Starting concurrency level.")
    parser.add_argument("--max-concurrency-limit", type=int, default=300, help="Maximum concurrency level to test.")
    parser.add_argument("--step", type=int, default=10, help="Step size to increase concurrency level.")
    parser.add_argument("--failure-threshold", type=float, default=0.2, help="Failure rate threshold to stop testing.")
    args = parser.parse_args()

    API_KEY = args.api_key
    START_CONCURRENCY = args.start_concurrency
    MAX_CONCURRENCY_LIMIT = args.max_concurrency_limit
    STEP = args.step
    FAILURE_THRESHOLD = args.failure_threshold

    logging.info(
        f"测试参数 - 初始并发数: {START_CONCURRENCY}, 最大并发数上限: {MAX_CONCURRENCY_LIMIT}, 步长: {STEP}, 失败率阈值: {FAILURE_THRESHOLD}")
    start_time = time.time()

    max_concurrency = asyncio.run(find_max_concurrency(
        API_KEY,
        START_CONCURRENCY,
        MAX_CONCURRENCY_LIMIT,
        STEP,
        FAILURE_THRESHOLD
    ))

    total_time = time.time() - start_time
    logging.info(f"测试完成，最大并发数: {max_concurrency}, 总耗时: {total_time:.2f} 秒")
    print(f"\n✅ 测试完成！API估计的最大并发数为: {max_concurrency}")
    print(f"总耗时: {total_time:.2f} 秒")


# 60
# 70开始有
# 130