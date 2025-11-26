import torch
# 修复1：使用 PyTorch 原生的 AdamW
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
# 修复2：只导入 load_dataset，不再导入 load_metric
from datasets import load_dataset
# 修复3：导入新的评估库
import evaluate
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 加载预训练的BERT分词器和模型
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2) 
model.to(device)

# 3. 加载轻量级文本分类数据集
dataset_name = "glue"
subset_name = "mrpc" 
raw_datasets = load_dataset(dataset_name, subset_name)

# 4. 数据预处理
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# 5. 创建数据加载器
batch_size = 16
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=batch_size)

# 6. 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 7. 加载评估指标 (修复点：使用 evaluate.load)
metric = evaluate.load("glue", subset_name)

# 8. 模型微调
epochs = 5
for epoch in range(epochs):
    print(f"\n======== Epoch {epoch + 1} / {epochs} ========")
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc="训练"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"  训练平均损失: {avg_train_loss:.4f}")

    model.eval()
    eval_loss = 0
    predictions = []
    references = []

    for batch in tqdm(eval_dataloader, desc="评估"):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            eval_loss += loss.item()
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            references.extend(labels.cpu().numpy())
    
    avg_eval_loss = eval_loss / len(eval_dataloader)
    print(f"  评估平均损失: {avg_eval_loss:.4f}")

    # 计算准确率
    results = metric.compute(predictions=predictions, references=references)
    print(f"  评估准确率: {results['accuracy']:.4f}")

print("\n微调完成！")