import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

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
        'It contains 60,000 32x32 color images in 10 different classes.\n')
    print(introduction)

    set_train = torchvision.datasets.CIFAR10('data', train = True, download = False) # 50000 images
    set_test = torchvision.datasets.CIFAR10('data', train = False, download = False) # 10000 images
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
    label_test = np.array(set_test.targets).astype(np.int32)

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



class CNN(nn.Module):
    def __init__(self, input_dim=(3, 32, 32), num_filters_1=6, num_filters_2=16, 
                 filter_size=5, hidden_dim=120, num_classes=10):
        super(CNN, self).__init__()
        
        # 計算每層輸出尺寸
        H, W = input_dim[1], input_dim[2]
        
        # 第一個卷積層後的尺寸
        H = H - filter_size + 1  # 32 - 5 + 1 = 28
        W = W - filter_size + 1  # 28
        
        # 第一個池化層後的尺寸
        H = H // 2  # 14
        W = W // 2  # 14
        
        # 第二個卷積層後的尺寸
        H = H - filter_size + 1  # 14 - 5 + 1 = 10
        W = W - filter_size + 1  # 10
        
        # 第二個池化層後的尺寸
        H = H // 2  # 5
        W = W // 2  # 5
        
        # 計算展平後的特徵數量
        fc_input_dim = num_filters_2 * H * W  # 16 * 5 * 5 = 400
        
        # 卷積層
        self.conv_layers = nn.Sequential(
            # 第一個卷積層塊
            nn.Conv2d(input_dim[0], num_filters_1, filter_size, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # 第二個卷積層塊
            nn.Conv2d(num_filters_1, num_filters_2, filter_size, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        # 全連接層
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 打印網絡結構信息
        print(f"Network architecture:")
        print(f"Input shape: {input_dim}")
        print(f"After conv1: {num_filters_1}x{H*2}x{W*2}")
        print(f"After pool1: {num_filters_1}x{H}x{W}")
        print(f"After conv2: {num_filters_2}x{H}x{W}")
        print(f"After pool2: {num_filters_2}x{H}x{W}")
        print(f"Flattened size: {fc_input_dim}")
        print(f"Hidden layer size: {hidden_dim}")
        print(f"Output size: {num_classes}")
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                k = 1.0 / (m.in_channels * m.kernel_size[0] * m.kernel_size[1])
                nn.init.uniform_(m.weight, -np.sqrt(k), np.sqrt(k))
            elif isinstance(m, nn.Linear):
                k = 1.0 / m.in_features
                nn.init.uniform_(m.weight, -np.sqrt(k), np.sqrt(k))
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -np.sqrt(k), np.sqrt(k))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    best_val_acc = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0 
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 添加維度檢查
            if i == 0:
                print(f"\nInput batch shape: {inputs.shape}")
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss/100:.4f}')
                train_losses.append(running_loss/100)
                running_loss = 0.0
        
        # 驗證階段
        val_acc = evaluate(model, val_loader, device)
        val_accuracies.append(val_acc)
        print(f'Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_accuracies

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def plot_training_history(train_losses, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Steps (x100)')
    ax1.set_ylabel('Loss')
    
    # Plot validation accuracy
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.show()

def main():
    # 獲取數據
    data = get_cifar10_data()
    
    # 創建數據加載器
    train_dataset = TensorDataset(
        torch.FloatTensor(data['X_train']), 
        torch.LongTensor(data['y_train'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['X_val']), 
        torch.LongTensor(data['y_val'])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(data['X_test']), 
        torch.LongTensor(data['y_test'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 初始化模型
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CNN().to(device)
    
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練模型
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, device
    )
    
    # 繪製訓練歷史
    plot_training_history(train_losses, val_accuracies)
    
    # 載入最佳模型並在測試集上評估
    model.load_state_dict(torch.load('best_model.pth'))
    test_acc = evaluate(model, test_loader, device)
    print(f'\nTest Accuracy: {test_acc:.2f}%')

if __name__ == '__main__':
    main()