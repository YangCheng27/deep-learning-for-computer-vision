import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm, trange
import time
import gc
from sklearn.metrics import confusion_matrix
import seaborn as sns


# use mps for GPU training acceleration
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Use mps for GPU training acceleration.")
    else: 
        device = torch.device("cpu")
        print("Use CPU device")
    return device



def get_cifar10_data(train_aug = True, visualization = True):
    if visualization: data_vis()

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.49139968, 0.48215827, 0.44653124],
            std = [0.24703233, 0.24348505, 0.26158768]
        )
    ])

    if train_aug:
        data_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.49139968, 0.48215827, 0.44653124],
                std = [0.24703233, 0.24348505, 0.26158768]
            )
        ])
    else:
        data_transform_train = data_transform

    dataset_train = torchvision.datasets.CIFAR10(
        root = 'data',
        train = True,
        download = False,
        transform = data_transform_train
    )
    dataset_test = torchvision.datasets.CIFAR10(
        root = 'data',
        train = False,
        download = False,
        transform = data_transform
    )
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset_train,
        [0.9, 0.1]
    )
    print(f'Training set size: {len(dataset_train)}')
    print(f'Validation set size: {len(dataset_val)}')
    print(f'Test set size: {len(dataset_test)}')

    dataloader_train = DataLoader(
        dataset_train,
        batch_size = 256,
        shuffle = True,
        num_workers = 6
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size = 256,
        shuffle = False,
        num_workers = 6
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size = 256,
        shuffle = False,
        num_workers = 6
    )

    return {
        'train_loader': dataloader_train,
        'val_loader': dataloader_val,
        'test_loader': dataloader_test,
        'classes': dataset_train.dataset.classes
    }
    


def data_vis():
    introduction = (
        'The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of\n'
        'images that are commonly used to train machine learning and computer vision algorithms.\n'
        'It contains 60,000 32x32 color images in 10 different classes.\n')
    print(introduction)

    set_train = torchvision.datasets.CIFAR10('data', train = True, download = False)
    class_names = set_train.classes
    print('Data set type: ', type(set_train))

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



class CNN(nn.Module):
    def __init__(self, input_dim = (3, 32, 32),
                 num_filters_conv1 = 64,
                 num_filters_conv2 = 128, 
                 num_filters_conv3 = 256, 
                 filter_size = 3,
                 hidden_dim = 1024,
                 num_classes = 10):
        super().__init__()

        C, H, W = input_dim
        padding_size = 1

        # conv1
        H_c1_1 = H + 2 * padding_size - filter_size + 1 # 32 + 2 - 3 + 1 = 32
        W_c1_1 = W + 2 * padding_size - filter_size + 1 # 32
        H_c1_2 = H_c1_1 + 2 * padding_size - filter_size + 1 # 32
        W_c1_2 = W_c1_1 + 2 * padding_size - filter_size + 1 # 32
        H_c1_m = H_c1_2 // 2 # 16
        W_c1_m = W_c1_2 // 2 # 16

        # conv2
        H_c2_1 = H_c1_m + 2 * padding_size - filter_size + 1 # 16 + 2 - 3 + 1 = 16
        W_c2_1 = W_c1_m + 2 * padding_size - filter_size + 1 # 16
        H_c2_2 = H_c2_1 + 2 * padding_size - filter_size + 1 # 16
        W_c2_2 = W_c2_1 + 2 * padding_size - filter_size + 1 # 16
        H_c2_m = H_c2_2 // 2 # 8
        W_c2_m = W_c2_2 // 2 # 8

        # conv3
        H_c3_1 = H_c2_m + 2 * padding_size - filter_size + 1 # 8 + 2 - 3 + 1 = 8
        W_c3_1 = W_c2_m + 2 * padding_size - filter_size + 1 # 8
        H_c3_2 = H_c3_1 + 2 * padding_size - filter_size + 1 # 8
        W_c3_2 = W_c3_1 + 2 * padding_size - filter_size + 1 # 8
        H_c3_m = H_c3_2 // 2 # 4
        W_c3_m = W_c3_2 // 2 # 4

        # fc layers
        fc_input_dim = num_filters_conv3 * H_c3_m * W_c3_m  # 256 * 4 * 4 = 400

        print(f"CNN architecture:")
        print(f"  Input shape: {input_dim}")
        print(f"  After conv1: {num_filters_conv1}x{H_c1_2}x{W_c1_2}")
        print(f"  After pool1: {num_filters_conv1}x{H_c1_m}x{W_c1_m}")
        print(f"  After conv2: {num_filters_conv2}x{H_c2_2}x{W_c2_2}")
        print(f"  After pool2: {num_filters_conv2}x{H_c2_m}x{W_c2_m}")
        print(f"  After conv3: {num_filters_conv3}x{H_c3_2}x{W_c3_2}")
        print(f"  After pool3: {num_filters_conv3}x{H_c3_m}x{W_c3_m}")
        print(f"  Flattened size: {fc_input_dim}")
        print(f"  Hidden layer size: {hidden_dim}")
        print(f"  Output size: {num_classes}\n")
        
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(C, num_filters_conv1, filter_size, padding = padding_size),
            nn.BatchNorm2d(num_filters_conv1),
            nn.ReLU(),
            nn.Conv2d(num_filters_conv1, num_filters_conv1, filter_size, padding = padding_size),
            nn.BatchNorm2d(num_filters_conv1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2), 
            
            # conv2
            nn.Conv2d(num_filters_conv1, num_filters_conv2, filter_size, padding = padding_size),
            nn.BatchNorm2d(num_filters_conv2),
            nn.ReLU(),
            nn.Conv2d(num_filters_conv2, num_filters_conv2, filter_size, padding = padding_size),
            nn.BatchNorm2d(num_filters_conv2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3), 
            
            # conv3
            nn.Conv2d(num_filters_conv2, num_filters_conv3, filter_size, padding = padding_size),
            nn.BatchNorm2d(num_filters_conv3),
            nn.ReLU(),
            nn.Conv2d(num_filters_conv3, num_filters_conv3, filter_size, padding = padding_size),
            nn.BatchNorm2d(num_filters_conv3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4), 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x



def train_model(model,
                train_aug,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                num_epochs = 5,
                patience = 6
                ):
    
    best_val_acc = 0
    best_epoch = 0
    no_improve = 0  
    best_model_params = None

    history = {
        'train_loss': [],
        'val_acc': [],
        'best_epoch': None,
        'best_acc': None,
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, 
                   desc=f'Epoch [{epoch+1}/{num_epochs}]',
                   ncols=100
                   )
        
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #scheduler.step()

            running_loss += loss.item()
            current_avg_loss = running_loss / (i + 1)
            pbar.set_postfix({
                'Avg Loss': f'{current_avg_loss:.4f}',
            })

        # training loss
        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)
        
        # validation
        val_acc = evaluate(model, val_loader, device)
        history['val_acc'].append(val_acc)
        
        #print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}')
        print(f'Val Acc: {val_acc:.2f}%')
        
        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_params = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            history['best_epoch'] = epoch
            history['best_acc'] = val_acc
            if train_aug:
                torch.save(model.state_dict(), 'best_model_aug.pth')
            else:
                torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved: {val_acc:.2f}%\n')
        else:
            no_improve += 1
            print()
            
        # Early stopping
        if no_improve >= patience:
            print(f'\nEarly stopping at epoch {epoch}')
            print(f'Best accuracy was {best_val_acc:.2f}% at epoch {best_epoch}')
            break

        if device.type == 'mps':
            gc.collect()
            torch.mps.empty_cache()

    if best_model_params is not None:
        model.load_state_dict(best_model_params)
    
    return history, model



def evaluate(model, data_loader, device, classes = None):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    if classes:
        print(f'Accuracy: {accuracy:.2f}%')

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, 
                   annot=True,      # disply number
                   fmt='d',         # int
                   cmap='Blues',
                   xticklabels=classes,
                   yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        metrics = get_metrics(cm)
        
        print("\nPer-class metrics:")
        print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        print("-" * 45)
        for i, class_name in enumerate(classes):
            m = metrics[i]
            print(f"{class_name:<10} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
        
        print("\nTop 5 Most Common Misclassifications:")
        misclassifications = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j:  # misclassification
                    misclassifications.append((cm[i][j], classes[i], classes[j]))
        
        # top 5 misclassifications
        misclassifications.sort(reverse=True)
        for count, true_class, pred_class in misclassifications[:5]:
            print(f"{true_class} classified as {pred_class}: {count} times")
    
    return accuracy



def get_metrics(confusion_matrix):
    metrics = {}
    n_classes = confusion_matrix.shape[0]
    
    for i in range(n_classes):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP
        
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        metrics[i] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics



def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot training loss
    ax1.plot(history['train_loss'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Steps (x100)')
    ax1.set_ylabel('Loss')
    
    # Plot validation accuracy
    ax2.plot(history['val_acc'])
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.show()

def main():

    data = get_cifar10_data(visualization=True)
    device = get_device()
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history, best_model = train_model(
        model=model,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=50
    )
    
    plot_training_history(history['train_loss'], history['val_acc'])
    
    model.load_state_dict(torch.load('best_model_aug.pth', weights_only=True))
    test_acc = evaluate(model, data['test_loader'], device, classes = data['classes'])
    print(f'\nFinal Test Accuracy: {test_acc:.2f}%')


if __name__ == '__main__':
    main()