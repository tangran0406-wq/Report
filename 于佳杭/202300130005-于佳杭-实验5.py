 import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os


class MRPCDataset(Dataset):
    def __init__(self, file_path, tokenizer=None, max_length=128):
        local_model_path = "./bert-base-uncased"
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained(local_model_path)
        self.max_length = max_length
        self.data = []
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                label = int(parts[0])
                sentence1 = parts[3]
                sentence2 = parts[4]
                self.data.append((sentence1, sentence2, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1, sentence2, label = self.data[idx]
        assert isinstance(sentence1, str), f"Expected string but got {type(sentence1)}"
        assert isinstance(sentence2, str), f"Expected string but got {type(sentence2)}"
        encoding = self.tokenizer(
            sentence1, sentence2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.float)
        }


class FCModel(nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def binary_accuracy(predictions, labels, threshold=0.5):
    binary_predictions = (predictions > threshold).float()
    correct = (binary_predictions == labels).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def train_model(bert_model, model, train_loader, crit, optimizer, bert_optimizer, device, num_epochs=3):
    bert_model.train()
    model.train()

    for epoch in range(num_epochs):
        epoch_loss, epoch_acc = 0., 0.
        total_len = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for i, data in enumerate(progress_bar):
            if torch.cuda.is_available():
                print(f"当前显存使用情况: {torch.cuda.memory_allocated(device)}")

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label = data['label'].to(device)
            label = torch.clamp(label, 0, 1)

            encoding = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

            bert_output = bert_model(**encoding)
            pooler_output = bert_output.pooler_output
            predict = model(pooler_output).squeeze()

            loss = crit(predict, label)
            acc = binary_accuracy(predict, label)

            optimizer.zero_grad()
            bert_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bert_optimizer.step()

            epoch_loss += loss.item() * len(label)
            epoch_acc += acc.item() * len(label)
            total_len += len(label)

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Accuracy': f'{acc.item():.4f}'
            })

        avg_loss = epoch_loss / total_len
        avg_acc = epoch_acc / total_len
        print(f'EPOCH {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')

    return bert_model, model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    local_model_path = "./bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    bert_model = BertModel.from_pretrained(local_model_path).to(device)
    model = FCModel().to(device)

    train_dataset = MRPCDataset('./msr_paraphrase_train.txt', tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=2e-5)

    print("开始训练...")
    bert_model, model = train_model(
        bert_model, model, train_loader, criterion,
        optimizer, bert_optimizer, device, num_epochs=3
    )

    torch.save({
        'bert_model_state_dict': bert_model.state_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bert_optimizer_state_dict': bert_optimizer.state_dict(),
    }, 'bert_mrpc_model.pth')

    print("训练完成，模型已保存!")


if __name__ == '__main__':
    main()