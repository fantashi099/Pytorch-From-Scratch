import torch
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from model import get_instance_segmentation_model

def display_image_in_actual_size(img):
    dpi = 80
    w, h = img.size
    figsize = w/ float(dpi), h/float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')

def plot_pred(y_pred, origin_img, savepath):
    for i in range(len(y_pred['boxes'])):
        # extract the confidence (probability) associated with the prediction
        confidence = y_pred['scores'][i]

        # filter out weak detections by ensuring the confidence is greather than threshold
        if confidence > 0.75:
            # extract the index of class label from the detections,
            # then compute the (x,y) of bbx for the object
            idx = int(y_pred['labels'][i])
            bbx = y_pred['boxes'][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = bbx.astype('int')
            classes = ['background', 'pedestrian']

            label = "{}: {:.2f}%".format(classes[idx], confidence*100)
            print("[INFO] {}".format(label))
            full_img = ImageDraw.Draw(origin_img)
            font = ImageFont.truetype("arial.ttf", 32)

            full_img.rectangle([(startX, startY), (endX, endY)], outline='blue')
            y = startY - 30 if startY - 30 > 30 else startY + 30
            full_img.text((startX, y), label, fill='blue', font=font)

    dpi = 80
    w, h = origin_img.size
    figsize = w/ float(dpi), h/float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')
    ax.imshow(origin_img)
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()

def normalize(img):
    # transpose [H,W,C] to [C, H, W]
    img = np.transpose(img, (2,0,1))
    # Add img to batch [1,C,H,W]
    img = np.expand_dims(img, axis=0)
    img = img/255.0
    img = torch.FloatTensor(img)
    return img

if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(description='Run Inference from trained model')
    parser.add_argument('--imagepath', type=str, required=True, default='./data/walking_people.jpg',
        help="Path to the input image")
    parser.add_argument('--modelpath', type=str, default='./ObjectDetection/final_model.bin', 
        help='Path to the trained model bin')
    parser.add_argument('--savepath', type=str, default='./ObjectDetection/result.jpg', 
        help='Path to the save figure')
    args = parser.parse_args()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_instance_segmentation_model(num_classes=2)
    model.load_state_dict(torch.load(args.modelpath))
    model.to(device)

    img = Image.open(args.imagepath).convert('RGB')
    origin_img = img.copy()
    normalize_img = normalize(img)
    model.eval()
    pred = model(normalize_img.to(device))[0]
    plot_pred(pred, origin_img, args.savepath)
