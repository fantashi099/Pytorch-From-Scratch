from engine import train_one_epoch, evaluate
from multiprocessing import freeze_support
from model import get_instance_segmentation_model
from torchvision import transforms
from data import get_data
import torch.optim as optim
import torch
from PIL import Image

def test_model(model, dataset, index, device):
    img, _ = dataset[index]
    model.eval()
    with torch.no_grad():
        y_pred = model([img.to(device)])
    # convert the image, which has been rescaled to 0-1 and 
    # had the channels flipped so that we have it in [C, H, W] format
    img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    # The masks are predicted as [N, 1, H, W], where N is the number of predictions, 
    # and are probability maps between 0-1.
    mask = Image.fromarray(y_pred[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
    return img, mask

def predict_img(model, path):
    img = Image.open(path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    # create a mini-batch as expected by the mode
    # input_batch = input_tensor.unsqueeze(0)
    pred = model([input_tensor])
    return pred

if __name__ == '__main__':
    freeze_support()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data()

    model = get_instance_segmentation_model(num_classes=2)
    model.to(device)

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.SGD(params, lr=3e-3, momentum=0.9, weight_decay=3e-4)

    # Learning rate scheduler with decreases the learning rate by 10x every 3 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(50):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # Update learning rate
        lr_scheduler.step()
        evaluate(model, test_loader, device=device)




