import torch
from PIL import Image
from multiprocessing import freeze_support
from model import get_instance_segmentation_model


if __name__ == '__main__':
    freeze_support()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_instance_segmentation_model(num_classes=2)
    model.load_state_dict(torch.load('./ObjectDetection/final_model.bin'))
    model.to(device)
    img = Image.open('./data/walking_people.jpg').convert('RGB')
    model.eval()
    with torch.no_grad():
        predict = model([img.to(device)])
    print(predict)