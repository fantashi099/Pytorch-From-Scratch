import torch
import torch.nn as nn
import numpy as np
from model import Model
from data import get_data
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

def train(epoch):
    model.train()
    history_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f})]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset),
                100*batch_idx/len(train_loader), loss.data.item()
            ))
        history_loss.append(loss.data.item())
    history_loss = np.array(history_loss)
    return np.mean(history_loss)

def test():
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            y_pred = model(data)
            total_loss += criterion(y_pred, target).data.item()
            pred = torch.max(y_pred, 1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    total_loss /= len(test_loader.dataset)
    print('\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f})\n'.format(
        total_loss, correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)
    ))
    return total_loss, 100*correct/len(test_loader.dataset)

if __name__ == '__main__':
    freeze_support()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data()

    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.5)

    h_acc, h_loss, h_val_loss = [], [], []
    for epoch in range(1, 20):
        loss = train(epoch)
        val_loss, val_acc  = test()
        h_acc.append(val_acc)
        h_loss.append(loss)
        h_val_loss.append(val_loss)
    
    x = np.arange(1,20)
    print(h_acc)
    plt.plot(x, h_loss, label='Train Loss')
    plt.plot(x, h_acc, label='Val Accuracy')
    plt.plot(x, h_val_loss, label='Val loss')
    plt.legend()
    plt.grid()
    plt.title('History train')
    plt.show()
