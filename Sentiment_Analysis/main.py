from sklearn.metrics import classification_report
import torch
import numpy as np
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from torch.optim import AdamW
from collections import defaultdict

from data import get_data
from model import SentimentClassifier

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)    
    total_loss = start_loss + end_loss
    return total_loss

def train():
    model.train()
    losses = []
    correct = 0

    for data in train_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_masks'].to(device)
        targets = data['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs, targets)
        _, pred = torch.max(outputs, dim=1)

        correct += torch.sum(pred == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

    print(f'Train Accuracy: {correct.double()/len(train_loader.dataset)} Loss: {np.mean(losses)}')
    return correct.double()/len(train_loader.dataset), np.mean(losses)


def test(test_data = False):
    model.eval()
    losses = []
    correct = 0

    with torch.no_grad():
        data_loader = test_loader if test_data else valid_loader
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_masks'].to(device)
            targets = data['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, pred = torch.max(outputs, dim=1)

            loss = criterion(outputs, targets)
            correct += torch.sum(pred == targets)
            losses.append(loss.item())
    
    if test_data:
        print(f'Test Accuracy: {correct.double()/len(test_loader.dataset)} Loss: {np.mean(losses)}')
        return correct.double()/len(test_loader.dataset), np.mean(losses)
    else:
        print(f'Valid Accuracy: {correct.double()/len(valid_loader.dataset)} Loss: {np.mean(losses)}')
        return correct.double()/len(valid_loader.dataset), np.mean(losses)

def eval(data_loader):
    model.eval()

    texts = []
    predicts = []
    predict_probs = []
    real_values = []

    with torch.no_grad():
        for data in data_loader:
            text = data['text']
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_masks'].to(device)
            targets = data['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, pred = torch.max(outputs, dim=1)
            texts.extend(text)
            predicts.extend(pred)
            predict_probs.extend(outputs)
            real_values.extend(targets)
    
    predicts = torch.stack(predicts).cpu()
    predict_probs = torch.stack(predict_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    print(classification_report(real_values, predicts))

def raw_predict(text, tokenizer, max_len=120):
    class_names = ['Enjoyment', 'Disgust', 'Sadness', 'Anger', 'Surprise', 'Fear', 'Other']
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, y_pred = torch.max(output, dim=1)

    print(f'Text: {text}')
    print(f'Sentiment: {class_names[y_pred]}')

if __name__ == '__main__':
    freeze_support()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    train_loader, valid_loader, test_loader = get_data(tokenizer)

    model = SentimentClassifier(7).to(device)
    criterion = nn.CrossEntropyLoss()

    # Recommendation by BERT: lr: 5e-5, 2e-5, 3e-5
    # Batchsize: 16, 32
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    history = defaultdict(list)
    best_acc = 0
    epochs = 10

    lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=0, 
                num_training_steps=len(train_loader)
            )

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-'*30)

        train_acc, train_loss = train()
        val_acc, val_loss = test()
        
        history['train_acc'].append(train_acc.item())
        history['train_loss'].append(train_loss.item())
        history['val_acc'].append(val_acc.item())
        history['val_loss'].append(val_loss.item())

        if val_acc > best_acc:
            torch.save(model.state_dict(), './Sentiment_Analysis/best_model.bin')
            best_acc = val_acc 
    
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    plt.title('Training History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0,1])
    plt.show()

    eval(test_loader)
    raw_predict('Cảm ơn bạn đã chạy thử model của mình. Chúc một ngày mới tốt lành nha!', tokenizer)