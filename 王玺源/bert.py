import torch
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
# ====================== 1. 核心配置 ======================
device = torch.device("cpu")
print(f"使用设备：{device}")
# 本地模型路径
MODEL_PATH = "./bert-base-chinese"
NUM_LABELS = 2  # 二分类
BATCH_SIZE = 4  # 进一步降低批次，适配CPU
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 64  # 缩短文本长度，加快CPU训练
# ====================== 2. 数据准备 ======================
def create_sample_data():
    data = {
        "text": [
            "这部电影太好看了，剧情紧凑，演员演技在线！",
            "难吃的要死，再也不会来这家餐厅了",
            "今天天气真好，心情特别棒",
            "快递速度太慢了，等了一周才到，差评",
            "这个手机性价比超高，推荐购买",
            "服务态度极差，体验感为零",
            "演唱会太燃了，全程嗨翻",
            "商品质量有问题，退货还很麻烦"
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, val_df
# 加载并转换数据集
train_df, val_df = create_sample_data()
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})
# ====================== 3. 数据预处理 ======================
# 加载本地分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
# 分词处理
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# 数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ====================== 4. 加载BERT模型 ======================
model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=NUM_LABELS
).to(device)

# ====================== 5. 训练参数 ======================
training_args = TrainingArguments(
    output_dir="./bert_sentiment_results",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    report_to="none",
    no_cuda=True,
    fp16=False,
)
# ====================== 6. 评估指标 ======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}
# ====================== 7. 训练模型 ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# 开始训练
print("\n===== 开始训练 =====")
trainer.train()
# ====================== 8. 模型预测 ======================
def predict_sentiment(text):
    # 预处理文本
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors="pt"
    ).to(device)
    # 推理模式
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    # 标签映射
    label_map = {0: "负面", 1: "正面"}
    return label_map[prediction]
# 测试预测
test_texts = [
    "这家店的菜品超好吃，下次还来！",
    "手机用了三天就坏了，售后还不处理，太坑了",
    "今天的会议很顺利，达成了预期目标"
]
print("\n===== 预测结果 =====")
for text in test_texts:
    sentiment = predict_sentiment(text)
    print(f"文本：{text}\n情感：{sentiment}\n")
# ====================== 9. 保存模型 ======================
model.save_pretrained("./bert_sentiment_model")
tokenizer.save_pretrained("./bert_sentiment_model")
print("模型已保存至 ./bert_sentiment_model")