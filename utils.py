import matplotlib.pyplot as plt
import torch 
import os 
import yaml
# ================ #
#  Utils
# ================ #
def unnormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)  # x = x * std + mean
    return img_tensor

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# ================ #
#  Evaluation Metrics
# ================ #
def compute_dice(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()

def compute_iou(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = ((pred + target) >= 1).float().sum(dim=(2, 3))

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# ================ #
#  Visualization
# ================ #
def plot_loss(losses, num_epochs, split='train', save_root='/data1/yoojinoh/codes/mlops/polyp-seg/output'):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), losses, marker='o')
    plt.title(f"{split} Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    os.makedirs(save_root, exist_ok=True)
    plt.savefig(f"{save_root}/{split}_loss.png")  
    plt.show()
    plt.close()

def plot_losses(train_losses, bce_losses, dice_losses, save_root='/data1/yoojinoh/codes/mlops/polyp-seg/output'):
    epochs = range(1, len(train_losses) + 1)
    os.makedirs(save_root, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Total Loss')
    plt.plot(epochs, bce_losses, label='BCE Loss')
    plt.plot(epochs, dice_losses, label='Dice Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 
    plt.savefig(f"{save_root}/train_val_losses.png")
    print("Loss plots saved to", save_root)

