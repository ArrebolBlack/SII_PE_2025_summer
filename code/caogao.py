# -------------- 高效评测函数 ---------------
# 计算单个样本的NDCG@k
# def calculate_ndcg_for_sample(predicted_list, ground_truth_item, k=10):
#     """
#     计算单个样本的NDCG@k
#     参数:
#     predicted_list: 模型预测的电影ID排序列表 [id1, id2, id3, ...]
#     ground_truth_item: 用户实际观看的下一部电影ID
#     k: NDCG@k中的k取值
#     返回:
#     ndcg: NDCG@k分数
#     """
#     # 截取前k个预测结果
#     predicted_list = predicted_list[:k]
#
#     # 计算相关性分数列表
#     relevance = []
#     for item_id in predicted_list:
#         if item_id == ground_truth_item:
#              relevance.append(1) # 相关项
#         else:
#             relevance.append(0) # 不相关项
#     # 计算DCG@k
#     dcg = 0
#     for i, rel in enumerate(relevance):
#         # 位置i的折损因子为log2(i+2)
#         discount = math.log2(i + 2)
#         dcg += rel / discount
#
#     # 计算IDCG@k
#     # 在本任务中，理想情况是将唯一相关项放在第一位
#     idcg = 1 / math.log2(1 + 1) # = 1
#
#     # 计算NDCG@k
#     if idcg > 0:
#         ndcg = dcg / idcg
#     else:
#         ndcg = 0
#
#     return ndcg
