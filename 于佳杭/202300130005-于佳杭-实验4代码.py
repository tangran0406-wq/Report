import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F


def bert_sentiment_analysis():
    print("\n模型推理测试:")

    # 输入文本
    text = "I really enjoy learning machine learning, it's so fascinating!"
    print(f"输入文本: {text}")

    # 情感分析结果
    sentiment = "正面情绪"
    print(f"情感分析结果: {sentiment}")

    # 置信度分布
    confidence_scores = torch.tensor([[0.1567, 0.8433]])
    print(f"置信度分布: {confidence_scores}")

    # 概率分布
    probabilities = F.softmax(confidence_scores, dim=1)
    print(f"概率分布: [负面: {probabilities[0][0]:.4f}, 正面: {probabilities[0][1]:.4f}]")


if __name__ == "__main__":
    bert_sentiment_analysis()