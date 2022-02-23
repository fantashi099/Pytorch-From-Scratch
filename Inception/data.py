import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from multiprocessing import freeze_support


def get_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 64

    train_set = torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', download=True, train=False, transform=transform)

    train_loader = data.DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=2)
    test_loader = data.DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=2)

    return train_loader, test_loader


if __name__ == '__main__':
    freeze_support()

    train_loader, test_loader = get_data()
    dataiter = iter(test_loader)
    image, label = next(dataiter)
    print(image.shape)