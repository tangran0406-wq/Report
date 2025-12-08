from transformers import BertTokenizer, BertModel

# 加载预训练BERT模型和分词器（bert-base-uncased为基础英文模型）
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 测试文本
text = "Hello, this is a simple BERT test."

# 分词并转换为模型输入格式
inputs = tokenizer(text, return_tensors='pt')  # 'pt'表示PyTorch张量（若用TensorFlow则为'tf'）

# 前向传播（获取BERT输出）
outputs = model(**inputs)

# 输出结果形状（验证模型运行正常）
print("BERT输出形状:", outputs.last_hidden_state.shape)