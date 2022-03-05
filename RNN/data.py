from torch.utils.data import Dataset, DataLoader
import csv

class NameDataset(Dataset):
    def __init__(self, is_train=False):
        path = './RNN/names_train.csv' if is_train else './RNN/names_test.csv'
        with open(path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        self.names = [row[0] for row in rows]
        self.labels = [row[1] for row in rows]
        self.len = len(self.labels)
        self.labels_list = list(sorted(set(self.labels)))

    def __getitem__(self, index):
        return self.names[index], self.labels[index]

    def __len__(self):
        return self.len

    def get_countries(self):
        return self.labels_list
    
    def get_country(self, id):
        return self.labels_list[id]

    def get_country_id(self, country):
        return self.labels_list.index(country)

if __name__ == '__main__':
    dataset = NameDataset(True)
    print(dataset.get_countries())
    print(dataset.get_country_id('Vietnamese'))

    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    print(len(train_loader.dataset))
    for idx, (name, countries) in enumerate(train_loader):
        print(idx, 'name', name, 'country', countries)