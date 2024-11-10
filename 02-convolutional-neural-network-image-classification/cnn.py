import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# use mps for GPU training acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Use mps for GPU training acceleration.")
else: 
    device = torch.device("cpu")
    print("Use CPU device")


def get_cifar10_data():

    introduction = (
        'The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of\n'
        'images that are commonly used to train machine learning and computer vision algorithms.\n'
        'It contains 60,000 32x32 color images in 10 different classes.')
    print(introduction)

    set_train = torchvision.datasets.CIFAR10('data', train = True) # 50000 images
    set_test = torchvision.datasets.CIFAR10('data', train = False) # 10000 images
    class_names = set_train.classes

    '''
    Visualization
    '''
    num_classes = len(class_names)
    num_examples = 10
    images_by_class = {i: [] for i in range(num_classes)}

    # get images for each class
    for img, label in set_train:
        if len(images_by_class[label]) < num_examples:
            images_by_class[label].append(img)
        if all(len(images) == num_examples for images in images_by_class.values()):
            break

    # Plot examples for each class
    plt.figure(figsize=(8, 8))
    for row in range(num_classes):
        for col in range(num_examples):
            ax = plt.subplot(num_classes, num_examples + 1, row * (num_examples + 1) + col + 2)
            img = images_by_class[row][col]
            ax.imshow(img)
            ax.axis("off")
        # Display class name at the start of each row
        plt.text(-420, 15, class_names[row], ha="center", va="center", fontsize=10, weight="bold")

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5)
    plt.axis('off')
    plt.show()

    '''
    Data preprocessing for training, validation and test
    '''
    # change the order of dimension and normalize the image
    # (N, H, W, C)  -> (N, C, H, W) for torch
    # [0, 255] -> [0, 1]
    # centered around zero (empirical value)
    data_train = (set_train.data.transpose(0, 3, 1, 2).astype(np.float32) / 256.) - 0.5
    data_test = (set_test.data.transpose(0, 3, 1, 2).astype(np.float32) / 256.) - 0.5
    label_train = np.array(set_train.targets).astype(np.int32)
    label_test = np.array(set_train.targets).astype(np.int32)

    data = {
        'classes': class_names,
        'X_train': data_train[:45000],
        'y_train': label_train[:45000],
        'X_val': data_train[45000:],
        'y_val': label_train[45000:],
        'X_test': data_test,
        'y_test': label_test,
    }

    return data
    







# class CNN(nn.Module):
#     """
#     This convolutional neural network architecture:
#     conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax
#     """
#     def __init__(self):


