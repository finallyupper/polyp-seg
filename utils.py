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

def config_to_tag(config):
    bs = config['default']['batch_size']
    lr = config['train']['lr']
    ep = config['train']['epochs']
    return f"bs{bs}_lr{lr:.0e}_ep{ep}"

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

def compute_precision(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    TP = (pred * target).sum(dim=(2, 3))
    FP = (pred * (1 - target)).sum(dim=(2, 3))

    precision = (TP + 1e-6) / (TP + FP + 1e-6)
    return precision.mean().item()

def compute_recall(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    TP = (pred * target).sum(dim=(2, 3))
    FN = ((1 - pred) * target).sum(dim=(2, 3))

    recall = (TP + 1e-6) / (TP + FN + 1e-6)
    return recall.mean().item()

def compute_accuracy(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    correct = (pred == target).float().sum(dim=(2, 3))
    total = torch.ones_like(target).sum(dim=(2, 3))

    acc = correct / total
    return acc.mean().item()

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

def visualize_predictions(images, masks, outputs, batch_idx, save_dir, num_samples=5):
    pred = torch.sigmoid(outputs)
    pred = (pred > 0.5).float()
    images_shown = 0

    for i in range(images.shape[0]):
        image = unnormalize(images[i].cpu()).permute(1, 2, 0).numpy()
        gt_mask = masks[i][0].cpu().numpy()
        pred_mask = pred[i][0].cpu().numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'pred_b{batch_idx}_{i}.png'))
        plt.close()

        images_shown += 1
        if images_shown >= num_samples:
            break
