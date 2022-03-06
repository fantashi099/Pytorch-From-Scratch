from model import RNNClasifier
import time
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import NameDataset

HIDDEN_SIZE = 100
N_LAYERS = 2
BATCH_SIZE = 32
N_EPOCHS = 50
N_CHARS = 128 #ASCII

test_dataset = NameDataset(False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

train_dataset = NameDataset(True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

N_COUNTRIES = len(train_dataset.get_countries())

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

def str2ascii_arr(msg):
    arr = [ord(c) for c in msg]
    return arr, len(arr)

def countries2tensor(countries):
    country_ids = [train_dataset.get_country_id(country) for country in countries]
    return torch.LongTensor(country_ids)

def pad_sequences(vectorized_seqs, seq_lengths, countries):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    
    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # Sort the target (countries) by seq_len
    target = countries2tensor(countries)
    if len(countries):
        target = target[perm_idx]
    
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor), \
        create_variable(seq_lengths), \
        create_variable(target)

def make_variable(names, countries):
    sequence_and_length = [str2ascii_arr(name) for name in names]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = [sl[1] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor(seq_lengths)
    return pad_sequences(vectorized_seqs, seq_lengths, countries)

def train():
    total_loss = 0

    for i, (names, countries) in enumerate(train_loader, 1):
        input, seq_lengths, target = make_variable(names, countries)
        output = classifier(input, seq_lengths)

        loss = criterion(output, target)
        total_loss += loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                epoch, i*len(names), len(train_loader.dataset),
                100*i*len(names) / len(train_loader.dataset),
                total_loss / i*len(names)
            ))
    return total_loss

def test(name=None):
    # Predict for a given name
    if name:
        input, seq_lengths, target = make_variable([name], [])
        output = classifier(input, seq_lengths)
        pred = output.data.max(1, keepdim=True)[1]
        country_id = pred.cpu().numpy()[0][0]
        print(name, 'is', train_dataset.get_country(country_id))
        return
    
    print('Evaluating trained model ...')
    correct = 0

    for names, countries in test_loader:
        input, seq_lengths, target = make_variable(names, countries)
        output = classifier(input, seq_lengths)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)
    ))


if __name__ == '__main__':
    classifier = RNNClasifier(N_CHARS, HIDDEN_SIZE, N_COUNTRIES, N_LAYERS)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), 'GPUs!')
        # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
        classifier = nn.DataParallel(classifier)

    if torch.cuda.is_available():
        classifier.cuda()
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(1, N_EPOCHS+1):
        train()
        test()
        test("Tien")
        test('Fantashi')
        test('Tommy')
        test('Vladimir')
        print('\nTime excute: {}\n' % time.time() - start)