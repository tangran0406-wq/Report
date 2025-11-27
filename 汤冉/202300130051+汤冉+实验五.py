import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from datasets import load_dataset
import evaluate
from tqdm import tqdm


# ================= 1. 自定义模型：BERT + 两层 MLP =================

class BertMRPCClassifier(nn.Module):
    def __init__(self, backbone_name: str = "bert-base-uncased",
                 hidden_size: int = 768,
                 mlp_hidden: int = 256,
                 num_labels: int = 2):
        super().__init__()
        # 预训练 BERT 主体
        self.bert = BertModel.from_pretrained(backbone_name)
        # 两层 MLP 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_labels)   # CrossEntropyLoss 里做 softmax
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output              # [batch, 768]
        logits = self.classifier(pooled)            # [batch, num_labels]

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# ================= 2. 文本预处理函数 =================

def build_tokenize_fn(tokenizer, max_len=128):
    def tokenize_batch(batch):
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
    return tokenize_batch


def main():
    # -------- 设备 --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # -------- 分词器 / 数据集 --------
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    raw_datasets = load_dataset("glue", "mrpc")

    tokenize_fn = build_tokenize_fn(tokenizer, max_len=128)
    encoded_datasets = raw_datasets.map(tokenize_fn, batched=True)

    # 只保留模型需要的字段
    encoded_datasets = encoded_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    encoded_datasets = encoded_datasets.rename_column("label", "labels")
    encoded_datasets.set_format("torch")

    train_ds = encoded_datasets["train"]
    valid_ds = encoded_datasets["validation"]

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    # -------- 模型 / 优化器 / 度量 --------
    model = BertMRPCClassifier(backbone_name=model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    metric = evaluate.load("glue", "mrpc")  # 会提供 accuracy, f1 等

    # -------- 训练循环 --------
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch + 1} / {num_epochs} =====")

        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc="训练"):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = out["loss"]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"  训练集平均损失: {avg_train_loss:.4f}")

        # --- 验证阶段 ---
        model.eval()
        eval_loss = 0.0
        all_preds = []
        all_refs = []

        for batch in tqdm(valid_loader, desc="验证"):
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}

                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = out["loss"]
                logits = out["logits"]

                eval_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_refs.extend(batch["labels"].cpu().numpy())

        avg_eval_loss = eval_loss / len(valid_loader)
        scores = metric.compute(predictions=all_preds, references=all_refs)

        print(f"  验证集平均损失: {avg_eval_loss:.4f}")
        print(f"  验证集准确率: {scores['accuracy']:.4f}")

    print("\n微调完成！")


if __name__ == "__main__":
    main()
