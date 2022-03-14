import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from gensim.utils import simple_preprocess
from torch.utils.data import Dataset, DataLoader

class SentimentDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=120):
        df = pd.read_excel(path, sheet_name=None)['Sheet1']
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        To customize dataset, inherit from Dataset class and implement
        __len__ & __getitem__
        __getitem__ should return 
            data:
                input_ids
                attention_masks
                text
                targets
        """
        row = self.df.iloc[index]
        text, label = self.get_input_data(row)

        # Encode_plus will:
        # (1) split text into token
        # (2) Add the '[CLS]' and '[SEP]' token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map token to their IDS
        # (5) Create attention mask
        # (6) Return a dictionary of outputs
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_masks': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long),
        }


    def labelencoder(self,text):
        if text=='Enjoyment':
            return 0
        elif text=='Disgust':
            return 1
        elif text=='Sadness':
            return 2
        elif text=='Anger':
            return 3
        elif text=='Surprise':
            return 4
        elif text=='Fear':
            return 5
        else:
            return 6

    def get_input_data(self, row):
        text = row['Sentence']
        text = ' '.join(simple_preprocess(text))
        label = self.labelencoder(row['Emotion'])

        return text, label

def get_data_loaders(path, tokenizer, batch_size, max_len=120):
    return DataLoader(
        SentimentDataset(path, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

def get_max_len(path1, path2, path3):
    data1 = pd.read_excel(path1, sheet_name=None)['Sheet1']
    data2 = pd.read_excel(path2, sheet_name=None)['Sheet1']
    data3 = pd.read_excel(path3, sheet_name=None)['Sheet1']

    all_data = data1.Sentence.tolist() + data2.Sentence.tolist() + data3.Sentence.tolist()
    all_data = [' '.join(simple_preprocess(text)) for text in all_data]
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    encoded_text = [tokenizer.encode(text, add_special_tokens=True) for text in all_data]
    
    # output = 164
    return max([len(text) for text in encoded_text])

def get_data(tokenizer):
    train_loader = get_data_loaders(path='./data/UIT-VSMEC/train_nor_811.xlsx', tokenizer=tokenizer, batch_size=16)
    valid_loader = get_data_loaders(path='./data/UIT-VSMEC/valid_nor_811.xlsx', tokenizer=tokenizer, batch_size=16)
    test_loader = get_data_loaders(path='./data/UIT-VSMEC/test_nor_811.xlsx', tokenizer=tokenizer, batch_size=16)

    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    train_loader, valid_loader, test_loader = get_data()
    data = next(iter(train_loader))
    print(data)
    print(data['input_ids'].shape)
    print(data['attention_masks'].shape)
    print(data['targets'].shape)