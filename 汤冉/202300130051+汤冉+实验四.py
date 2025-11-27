import torch
from transformers import BertTokenizer, BertForSequenceClassification


class BertSmokeTest:
    """
    对 BERT 环境做一个简单的“冒烟测试”：
    1. 加载分词器
    2. 加载序列分类模型
    3. 编码一小段文本
    4. 做一次前向传播
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def show_header(self):
        print("=" * 50)
        print(f"BERT 环境检查  (model = {self.model_name})")
        print(f"当前设备: {self.device}")
        print("=" * 50)

    def load_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        print("[1/4] 分词器加载成功")

    def load_model(self):
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        self.model.to(self.device)
        print("[2/4] 模型加载成功，并已移动到设备")

    def encode_example(self):
        assert self.tokenizer is not None, "请先加载分词器"

        sentences = [
            "Hello, this is a test sentence.",
            "BERT is checking whether everything works."
        ]
        encodings = self.tokenizer(
            sentences,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64
        )
        print("[3/4] 示例文本编码完成")
        print(f"      input_ids 形状:     {tuple(encodings['input_ids'].shape)}")
        print(f"      attention_mask 形状:{tuple(encodings['attention_mask'].shape)}")
        return encodings

    def forward_once(self, encodings):
        assert self.model is not None, "请先加载模型"

        self.model.eval()
        batch = {k: v.to(self.device) for k, v in encodings.items()}
        # 构造一组假标签
        fake_labels = torch.zeros(batch["input_ids"].size(0), dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(**batch, labels=fake_labels)

        print("[4/4] 前向传播正常完成")
        print(f"      logits 形状: {tuple(outputs.logits.shape)}")
        print(f"      loss 数值:   {float(outputs.loss):.4f}")

    def run(self):
        self.show_header()
        self.load_tokenizer()
        self.load_model()
        enc = self.encode_example()
        self.forward_once(enc)
        print("-" * 50)
        print("检查结束：BERT 相关依赖和模型运行正常。")


if __name__ == "__main__":
    tester = BertSmokeTest("bert-base-uncased")
    tester.run()
