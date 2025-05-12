import json
import requests
import openai
import pandas as pd
from tqdm import tqdm
import anthropic  # Claude SDK

# ========== 配置 ==========
YOUR_NAME = "殷家琦"
YOUR_ID = "2022110714"
TESTSET_NAME = "final_test_dataset"
MODEL_NAMES = ["DeepSeek", "OpenAI", "Claude"]
NUM_QUESTIONS = 20

# ✅ API 密钥
DEEPSEEK_API_KEY = "sk-60981d4439e34af29f9b9afef32d6c7e"
OPENAI_API_KEY = "sk-iHQc2a43268e1488cc9a51899e81fb3ac49557c106fsnOdb"
CLAUDE_API_KEY = "sk-iHQc2a43268e1488cc9a51899e81fb3ac49557c106fsnOdb"

# ✅ API base_url（如使用 GPTSAPI 代理）
OPENAI_API_BASE = "https://api.gptsapi.net/v1"
CLAUDE_API_BASE = "https://api.gptsapi.net"

# ========== 加载测试题 ==========
with open("final_test_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
dataset = dataset[:NUM_QUESTIONS]

# ========== 初始化 SDK ==========
openai_client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE
)

claude_client = anthropic.Anthropic(
    api_key=CLAUDE_API_KEY,
    base_url=CLAUDE_API_BASE
)

# ========== 模型调用函数 ==========

def call_deepseek(question_text):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",  # ✅ 使用 reasoner 模型
        "messages": [{"role": "user", "content": question_text}],
        "temperature": 0.2
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"

def call_openai(question_text):
    try:
        response = openai_client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[{"role": "user", "content": question_text}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"

def call_claude(question_text):
    try:
        message = claude_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=512,
            temperature=0.2,
            messages=[{"role": "user", "content": question_text}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"

# ========== 判断是否正确 ==========
def is_correct(pred, gold):
    pred = pred.strip().lower()
    gold = gold.strip().lower()
    return gold in pred or pred in gold

# ========== 执行测试 ==========
results = []
score_summary = {model: 0 for model in MODEL_NAMES}

for idx, item in enumerate(dataset):
    question = item["question"]
    answer = item["answer"]
    options = item.get("options", [])
    is_custom = "自编" if item["source"] == "自编" else "挑选"

    print(f"🔍 正在测试第 {idx + 1} 题：{question[:30]}...")

    row = {
        "姓名": YOUR_NAME,
        "学号": YOUR_ID,
        "测试集": TESTSET_NAME,
        "题号": idx + 1,
        "问题": question,
        "中文译文": question if all(ord(c) < 128 for c in question) else "",
        "标准答案": answer,
        "是否自编": is_custom
    }

    # DeepSeek
    deepseek_response = call_deepseek(question)
    row["DeepSeek回答"] = deepseek_response
    result = "✅" if is_correct(deepseek_response, answer) else "❌"
    row["DeepSeek判定"] = result
    if result == "✅":
        score_summary["DeepSeek"] += 1

    # OpenAI
    openai_response = call_openai(question)
    row["OpenAI回答"] = openai_response
    result = "✅" if is_correct(openai_response, answer) else "❌"
    row["OpenAI判定"] = result
    if result == "✅":
        score_summary["OpenAI"] += 1

    # Claude
    claude_response = call_claude(question)
    row["Claude回答"] = claude_response
    result = "✅" if is_correct(claude_response, answer) else "❌"
    row["Claude判定"] = result
    if result == "✅":
        score_summary["Claude"] += 1

    results.append(row)

# ========== 保存表格 ==========
df = pd.DataFrame(results)
df.to_excel("模型答题对比分析.xlsx", index=False)
df.to_csv("模型答题对比分析.csv", index=False)
print("✅ 测试结果已保存为：模型答题对比分析.xlsx")

# ========== 输出 Markdown 报告 ==========
with open("测试报告.md", "w", encoding="utf-8") as f:
    f.write(f"# 模型测试报告\n\n")
    f.write(f"- 姓名：{YOUR_NAME}\n")
    f.write(f"- 学号：{YOUR_ID}\n")
    f.write(f"- 测试集：{TESTSET_NAME}\n")
    f.write(f"- 测试题目数量：{NUM_QUESTIONS} 道题\n\n")

    f.write("## ✅ 模型正确率统计：\n\n")
    for model in MODEL_NAMES:
        acc = score_summary[model]
        f.write(f"- {model}：{acc}/{NUM_QUESTIONS} 准确率：{acc / NUM_QUESTIONS:.2%}\n")
    
    f.write("\n---\n## 📄 详细题目结果请见 Excel 文件：`模型答题对比分析.xlsx`\n")

print("📄 Markdown 报告已生成：测试报告.md")
