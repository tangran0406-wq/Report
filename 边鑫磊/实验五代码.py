import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate
import numpy as np

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 1. 加载MRPC数据集
print("加载MRPC数据集...")
dataset = load_dataset("glue", "mrpc")

# 2. 加载预训练模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

print("预处理数据...")
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. 数据收集器（用于动态填充）
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. 加载评估指标（MRPC使用准确率和F1分数）
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels)
    return {**acc, **f1_score}

# 6. 设置训练参数 - 修改了evaluation_strategy为eval_strategy
training_args = TrainingArguments(
    output_dir="./mrpc-bert-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",  # 注意这里改为eval_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# 7. 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8. 训练模型
print("开始训练模型...")
trainer.train()

# 9. 评估模型
print("评估模型...")
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")

# 10. 在测试集上评估
print("在测试集上评估...")
test_results = trainer.evaluate(tokenized_dataset["test"])
print(f"测试集结果: {test_results}")

# 11. 保存模型
print("保存模型...")
trainer.save_model("./mrpc-bert-final")

# 12. 模型推理示例
print("\n模型推理示例:")
def predict_paraphrase(sentence1, sentence2):
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "是释义" if predicted_class == 1 else "不是释义"

# 测试几个例子
examples = [
    ("The cat sat on the mat.", "A cat was sitting on the mat."),
    ("I love programming.", "Programming is my passion."),
    ("The weather is nice today.", "It's raining heavily outside.")
]

for s1, s2 in examples:
    result = predict_paraphrase(s1, s2)
    print(f"句子1: {s1}")
    print(f"句子2: {s2}")
    print(f"预测结果: {result}\n")