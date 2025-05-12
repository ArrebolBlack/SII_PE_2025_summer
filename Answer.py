import json
import re


def construct_prompt(d): # 0.7306
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


def parse_output(text):
    """
    更健壮的解析函数，自动识别中英文逗号、空格、换行分隔的 ID 序列
    """
    # logging.info(f"模型输出解析 - 原始输出:\n{text}")

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

    # logging.info(f"模型输出解析 - 解析后电影ID列表: {parsed_list}")
    return parsed_list