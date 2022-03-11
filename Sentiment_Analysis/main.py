import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

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
        attention_mask = data['attention_mask'].to(device)
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

    print(f'Train loss: {correct.double()/len(train_loader.dataset)} Accuracy: {np.mean(losses)}')
    return correct.double()/len(train_loader.dataset), np.mean(losses)


def test(test_data = False):
    model.eval()
    losses = []
    correct = 0

    with torch.no_grad():
        data_loader = test_loader if test_data else valid_loader
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
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
        print(f'Test loss: {correct.double()/len(train_loader.dataset)} Accuracy: {np.mean(losses)}')
    else:
        print(f'Valid loss: {correct.double()/len(train_loader.dataset)} Accuracy: {np.mean(losses)}')
    return correct.double()/len(train_loader.dataset), np.mean(losses)


if __name__ == '__main__':
    freeze_support()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, valid_loader, test_loader = get_data()

    model = SentimentClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    # Recommendation by BERT: lr: 5e-5, 2e-5, 3e-5
    # Batchsize: 16, 32
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    
    history = {}
    best_acc = 0
    epochs = 10
    
    lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=0, 
                num_training_steps=len(train_loader)*epochs
            )

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-'*30)

        train_acc, train_loss = train()
        val_acc, val_loss = test()
        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_acc:
            torch.save(model.state_dict(), './Sentiment_Analysis/best_model.bin')
            best_acc = val_acc 
        
    print(history['train_acc'])
    print(history['train_loss'])
    print(history['val_acc'])
    print(history['val_loss'])
    val_acc, val_loss = test(True)