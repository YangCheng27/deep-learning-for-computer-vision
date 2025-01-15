import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
import kagglehub
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Use mps for GPU training acceleration.")
    else: 
        device = torch.device("cpu")
        print("Use CPU device")
    return device


class CovidCTDataset(Dataset):
    def __init__(self, dataset_path, transform, train = True):
        self.dataset_path = dataset_path
        self.transform = transform
        self.train = train

        self.images_dir = os.path.join(dataset_path, 'frames')
        self.masks_dir = os.path.join(dataset_path, 'masks')

        self.image_files = sorted(os.listdir(self.images_dir))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        #img = torch.from_numpy(img).permute(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        if self.train:
            img, mask = self.transform(img, mask)
        else:
            img = transforms.functional.resize(img, (224, 224))
            mask = transforms.functional.resize(mask, (224, 224))
            
        return img, mask
    

class TrainTransform:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image, mask):
        # 調整大小
        image = transforms.functional.resize(image, (self.size, self.size))
        mask = transforms.functional.resize(mask, (self.size, self.size))
        
        if random.random() > 0.5:
            # 添加高斯噪聲
            noise = torch.randn_like(image) * 0.1
            image = image + noise
            image = torch.clamp(image, 0, 1)
        # 水平翻轉
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
            
        # 垂直翻轉
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        
        # 旋轉（90度的倍數）
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)
            
        # 亮度對比度調整（只對輸入圖像做）
        # if random.random() > 0.5:
        #     brightness_factor = random.uniform(0.8, 1.2)
        #     contrast_factor = random.uniform(0.8, 1.2)
        #     image = transforms.functional.adjust_brightness(image, brightness_factor)
        #     image = transforms.functional.adjust_contrast(image, contrast_factor)
        
        # 標準化（只對輸入圖像）
        # image = transforms.functional.normalize(image, 
        #                    mean=[0.485, 0.456, 0.406],
        #                    std=[0.229, 0.224, 0.225])
        
        return image, mask


def get_dataloaders(path, batch_size = 8, num_workers = 8):
    train_transform = TrainTransform(size = 224)

    train_dataset = CovidCTDataset(path, transform = train_transform)
    val_dataset = CovidCTDataset(path, transform = None, train = False)
    test_dataset = CovidCTDataset(path, transform = None, train = False)

    train_loader = DataLoader(train_dataset, 
                            batch_size = batch_size, 
                            shuffle = True, 
                            num_workers = num_workers)
    
    val_loader = DataLoader(val_dataset, 
                           batch_size = batch_size, 
                           shuffle = False, 
                           num_workers = num_workers)
    
    test_loader = DataLoader(test_dataset, 
                            batch_size = batch_size, 
                            shuffle = False, 
                            num_workers = num_workers)

    return {
        'training': train_loader,
        'evaluation': val_loader,
        'test': test_loader
    }


def data_vis(loader, num_samples=10):
    """
    顯示驗證集中的樣本和對應的遮罩
    Args:
        loader: data loader
        num_samples: 要顯示的樣本數量（預設為10）
    """

    val_dataset = loader.dataset
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))
    
    for idx in range(num_samples):
        img, mask = val_dataset[idx * 30]
        
        #axes[0, idx].imshow(image.permute(1, 2, 0))
        axes[0, idx].imshow(img.squeeze(), cmap='gray')
        axes[0, idx].axis('off')
        axes[0, idx].set_title(f'Image {idx+1}')
        
        axes[1, idx].imshow(mask.squeeze())
        axes[1, idx].axis('off')
        axes[1, idx].set_title(f'Mask {idx+1}')
    
    plt.tight_layout()
    plt.show()


class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x
    

class SimpleUNet2(nn.Module):
    def __init__(self):
        super(SimpleUNet2, self).__init__()
        
        # 1. 添加 BatchNorm
        # 2. 增加特徵通道
        # 3. 使用 skip connections
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 256 因為有 skip connection
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 128 因為有 skip connection
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)  # 最後用 1x1 卷積
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        # Middle
        x = self.middle(x)
        
        # Decoder with skip connections
        x = self.up1(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec2(x)
        
        return x

class SegmentationMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
    
    # Compare results
    def update(self, pred, target):
        if torch.is_tensor(pred): pred = pred.detach()
        if torch.is_tensor(target): target = target.detach()
            
        # ensure data formats are matching
        pred = (pred > 0.5).float()
        target = target.float()
        
        # get metrics result
        dice = self.calculate_dice(pred, target)
        iou = self.calculate_iou(pred, target)
        
        # move back to CPU for simple computation
        self.dice_scores.append(dice.cpu().item())
        self.iou_scores.append(iou.cpu().item())

    @staticmethod
    def calculate_dice(pred, target):
        smooth = 1e-5
        intersection = torch.sum(pred * target)
        return (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    
    @staticmethod
    def calculate_iou(pred, target):
        smooth = 1e-5
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        return (intersection + smooth) / (union + smooth)
    
    @staticmethod
    def calculate_sensitivity_specificity(pred, target):
        smooth = 1e-5
        
        pred = pred > 0.5
        target = target > 0.5
        
        TP = torch.sum((pred & target).float())
        TN = torch.sum((~pred & ~target).float())
        FP = torch.sum((pred & ~target).float())
        FN = torch.sum((~pred & target).float())
        
        sensitivity = (TP + smooth) / (TP + FN + smooth)
        specificity = (TN + smooth) / (TN + FP + smooth)
        
        return sensitivity, specificity
    
    def get_metrics(self):
        return {
            'Dice Score': np.mean(self.dice_scores),
            'IoU': np.mean(self.iou_scores),
        }


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * target)
        dice_loss = 1 - (2. * intersection + self.smooth) / (torch.sum(pred) + torch.sum(target) + self.smooth)
        return dice_loss
    

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce_loss(pred, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
    

def train_model(model, train_loader, val_loader, num_epochs = 10, device = None):
    if device == None: get_device()
    model = model.to(device)

    #criterion = CombinedLoss(dice_weight=0.65, bce_weight=0.35)
    #optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    criterion = CombinedLoss(dice_weight=0.7, bce_weight=0.3)
    # 使用 AdamW 優化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 使用 CosineAnnealingWarmRestarts 學習率調度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    # 添加早停機制
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    best_model_state = None

    train_metrics = SegmentationMetrics()
    val_metrics = SegmentationMetrics()

    train_losses = []
    val_losses = []
    epoch_times = []
    train_metrics_history = []
    val_metrics_history = []

    print(f"\nStart to train using: {device}")

    #for epoch in range(1, num_epochs + 1):
    for epoch in tqdm(range(1, num_epochs + 1), desc='Training Epochs'):
        epoch_start = time.time()

        model.train()
        epoch_loss = 0
        batch_times = []
        train_metrics.reset()

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=False)
        for batch_idx, (imgs, masks) in enumerate(train_pbar):
        #for batch_idx, (imgs, masks) in enumerate(train_loader):
            batch_start = time.time()
            imgs = imgs.to(device)
            masks = masks.to(device)
           
            optimizer.zero_grad()
            outputs = model(imgs)
            
            predictions = torch.sigmoid(outputs)
            train_metrics.update(predictions, masks)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            batch_end = time.time()
            batch_times.append(batch_end - batch_start)

            train_pbar.set_postfix({'batch_loss': '{:.4f}'.format(loss.item())})

            # if (batch_idx + 1) % 10 == 0:
            #     print(f'Batch {batch_idx + 1} / {len(train_loader)}, '
            #           f'Average batch time: {np.mean(batch_times[-10:]):.3f}s')
        

        model.eval()
        val_loss = 0
        val_metrics.reset()

        val_pbar = tqdm(val_loader, desc='Validation', leave=False)
        with torch.no_grad():
            #for images, masks in val_loader:
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                predictions = torch.sigmoid(outputs)
                val_metrics.update(predictions, masks)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

        


        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)

        train_metrics_history.append(train_metrics.get_metrics())
        val_metrics_history.append(val_metrics.get_metrics())


        print(f'\nEpoch {epoch}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Epoch Time: {epoch_time:.2f}s')
        
        train_results = train_metrics.get_metrics()
        val_results = val_metrics.get_metrics()
        
        print("\nTraining Metrics:")
        for metric, value in train_results.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nValidation Metrics:")
        for metric, value in val_results.items():
            print(f"{metric}: {value:.4f}")

        if (epoch) % 3 == 0:
            visualize_results(model, val_loader, device)

        # 更新學習率
        scheduler.step()
        
        # 早停檢查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                model.load_state_dict(best_model_state)
                break

    return train_losses, val_losses, epoch_times, train_metrics_history, val_metrics_history


def visualize_results(model, val_loader, device, num_samples = 4):
    model.eval()
    images, masks = next(iter(val_loader))
    images = images.to(device)
    masks = masks
    
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.sigmoid(outputs) > 0.5
    
    fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))
    
    for i in range(num_samples):
        idx = i * 2
        axes[0, i].imshow(images[idx,0].cpu(), cmap='gray')
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(masks[idx,0])
        axes[1, i].set_title(f'True Mask {i+1}')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(predictions[idx,0].cpu())
        axes[2, i].set_title(f'Predict Mask {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_result(train_losses, val_losses, train_metrics_history, val_metrics_history):
    
    epochs = range(1, len(train_metrics_history) + 1)
    metrics = train_metrics_history[0].keys()  # 指標名稱列表
    
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    for i, metric in enumerate(metrics, 2):
        train_values = [m[metric] for m in train_metrics_history]
        val_values = [m[metric] for m in val_metrics_history]
        
        plt.subplot(1, 3, i)
        plt.plot(epochs, train_values, label='Train')
        plt.plot(epochs, val_values, label='Validation')
        plt.title(f'{metric} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return


def plot_training_metrics(train_losses, val_losses, epoch_times):
    """繪製訓練指標"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 損失曲線
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 時間統計
    ax2.plot(epoch_times, marker='o')
    ax2.set_title('Epoch Training Times')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.show()


def plot_metrics_history(train_metrics_history, val_metrics_history):
    """視覺化訓練和驗證過程中的各種指標變化"""
    epochs = range(1, len(train_metrics_history) + 1)
    metrics = train_metrics_history[0].keys()  # 指標名稱列表
    
    plt.figure(figsize=(18, 12))
    for i, metric in enumerate(metrics, 1):
        train_values = [m[metric] for m in train_metrics_history]
        val_values = [m[metric] for m in val_metrics_history]
        
        plt.subplot(3, 2, i)
        plt.plot(epochs, train_values, label='Train')
        plt.plot(epochs, val_values, label='Validation')
        plt.title(f'{metric} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, val_loader, num_samples=3):
    """視覺化分割結果並顯示評估指標"""
    model.eval()
    metrics = SegmentationMetrics()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5
            
            # 更新指標
            metrics.update(predictions, masks)
            
            # 獲取當前樣本的指標
            sample_metrics = metrics.get_metrics()
            
            # 顯示圖像
            axes[i, 0].imshow(images[0,0].cpu(), cmap='gray')
            axes[i, 0].set_title('Original Image')
            
            axes[i, 1].imshow(masks[0,0].cpu(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            
            axes[i, 2].imshow(predictions[0,0].cpu(), cmap='gray')
            axes[i, 2].set_title('Prediction')
            
            # 顯示指標
            metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in sample_metrics.items()])
            axes[i, 3].text(0.1, 0.5, metrics_text, fontsize=10)
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()


