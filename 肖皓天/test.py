import torch
from transformers import BertTokenizer, BertForSequenceClassification

def test_bert_model_import():
    print("--- 开始测试 BERT 模型导入 ---")

    # 1. 尝试加载 BERT 分词器
    model_name = "bert-base-uncased"
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        print(f"   成功加载 BERT 分词器: {model_name}")
    except Exception as e:
        print(f"   载入 BERT 分词器失败: {e}")
        return

    # 2. 尝试加载 BERT 序列分类模型
    try:
        # 假设为二分类任务，这里num_labels的值不影响导入本身
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        print(f"   成功加载 BERT 序列分类模型: {model_name}")
        
        # 3. 尝试将模型移动到设备 (如果CUDA可用)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"模型已成功移动到设备: {device}")
        
    except Exception as e:
        print(f"导入 BERT 序列分类模型失败: {e}")
        return

    # 4. 尝试使用分词器对文本进行编码
    test_text = "Hello, this is a test sentence for BERT model."
    try:
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        print(f"   成功使用分词器编码文本: '{test_text}'")
        print(f"   Input IDs shape: {inputs['input_ids'].shape}")
        print(f"   Attention Mask shape: {inputs['attention_mask'].shape}")
    except Exception as e:
        print(f"文本编码失败: {e}")
        return
        
    # 5. 尝试将编码后的输入传递给模型 (进行一次前向传播)
    try:
        # 将输入移动到模型所在的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        dummy_labels = torch.tensor([0]).to(device) # 假设标签为0
        outputs = model(**inputs, labels=dummy_labels)
        
        print(f"   成功将编码输入传递给模型并获取输出。")
        print(f"   模型输出 logits shape: {outputs.logits.shape}")
        print(f"   模型输出 loss: {outputs.loss.item():.4f}") # 打印一下loss
        
    except Exception as e:
        print(f"   模型前向传播失败: {e}")
        return

    print("--- BERT 模型导入测试完成 ---")
    print("所有测试步骤均成功！ BERT 环境已准备就绪。")

if __name__ == "__main__":
    test_bert_model_import()