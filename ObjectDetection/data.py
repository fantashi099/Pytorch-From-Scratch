import os
import numpy as np
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from PIL import Image
import transforms as T
import utils

# https://discuss.pytorch.org/t/dataloader-gives-stack-expects-each-tensor-to-be-equal-size-due-to-different-image-has-different-objects-number/91941/7
# https://discuss.pytorch.org/t/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-3-224-224-at-entry-0-and-3-224-336-at-entry-3/87211/10

class PennFudanDataset(Dataset):
    """
    To customize dataset, inherit from Dataset class and implement
    __len__ & __getitem__
    __getitem__ should return 
        image: PIL image (H,W)
        target:
            boxes (FloatTensor[N,4]): [x0,x1,x2,x3] (0 to W, 0 to H)
            labels (Int64Tensor[N]): the label for each bbx, 0 = background
            image_id (Int64Tensor[1]): Image Identifier. Is used during evaluation
            area (Tensor[N]): The area of bbx. Is used during evaluation 
                (separate metric scores small-medium-large)
            iscrowd (UInt8Tensor[N]): True = ignore during evaluation
            (option) masks (UInt8Tensor[N,H,W]): Segmentation masks for each obj
            (opt) keypoints (FloatTensor[N,K,3]): For each N obj, it contain K keypoints
                [X, y, visibility] define the obj, visibility = 0 ~ keypoint is not visible.
                For augmentation, keypoint dependent on data representation
    """
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms
        # Load image & sort to ensure they are aligned (1 -> n)
        self.imgs = list(sorted(os.listdir(os.path.join(path, 'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(path, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.path, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert('RGB')
        # Each color of mask ~ different instance
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # First id = background -> remove it
        obj_ids = obj_ids[1:]

        # masks = [[[0 .... 1 1 1 1 .... 2 2 2 ... 0 0]]]
        # assume idx = 0 have 2 instances
        # split color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None] # ~ reshape obj_ids ([1,2]) to [[[1]],[[2]]]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for index in range(num_objs):
            pos = np.where(masks[index])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances aren't crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks 
        target['image_id'] = image_id
        target['area'] = area 
        target['iscrowd'] = iscrowd 

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transforms(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_data():
    dataset = PennFudanDataset("./data/PennFudanPed/", get_transforms(True))
    testset = PennFudanDataset("./data/PennFudanPed/", get_transforms(False))
    torch.manual_seed(86)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = Subset(dataset, indices[:-25])
    testset = Subset(testset, indices[-25:])

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    test_loader = DataLoader(testset, batch_size=16, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    return data_loader, test_loader

if __name__ == '__main__':
    dataset = PennFudanDataset("./data/PennFudanPed/")
    print(dataset[0])