from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
from dataset import get_data_loader 
import torch 
import matplotlib.pyplot as plt
from tqdm import tqdm 
from utils import plot_losses, compute_dice, compute_iou, unnormalize
import os 
from utils import load_yaml
import argparse

NUM_SAMPLES = 5

# Loss
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def bce_dice_loss(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return {
        'loss': bce + dice,
        'bce_loss': bce,
        'dice_loss': dice
    }


def train(num_epochs, train_loader, val_loader, model, optimizer, criterion, device, ckpt_save_dir='/data1/yoojinoh/codes/mlops/ckpt',):
    train_losses = [] 
    bce_losses = []
    dice_losses = []
    best_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        bce_loss = 0.0
        dice_loss = 0.0

        for idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']

            losses = criterion(outputs, masks)
            loss = losses['loss']

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            bce_loss += losses['bce_loss'].item() * images.size(0)
            dice_loss += losses['dice_loss'].item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        avg_bce_loss = bce_loss / len(train_loader.dataset)
        avg_dice_loss = dice_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        bce_losses.append(avg_bce_loss)
        dice_losses.append(avg_dice_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f} (BCE: {avg_bce_loss:.4f}, Dice: {avg_dice_loss:.4f})")


        val_dice = validate(model, 
                val_loader, 
                criterion, 
                device, 
                save_pred=False)

        if best_dice < val_dice:
            best_dice = val_dice
            best_model_path = os.path.join(ckpt_save_dir, 'deeplabv3_best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")

        if epoch == (num_epochs - 1):
            os.makedirs(ckpt_save_dir, exist_ok=True)
            save_name = f'deeplabv3_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), os.path.join(ckpt_save_dir, save_name))
            print(f"Last Model saved to {os.path.join(ckpt_save_dir, save_name)}")

    return model, train_losses, bce_losses, dice_losses

def validate(model, val_loader, criterion, device, save_pred=False, save_dir='/data1/yoojinoh/codes/mlops/polyp-seg/output'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    val_loss = 0.0
    dice_total = 0.0
    iou_total = 0.0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)['loss']
            val_loss += loss.item() * images.size(0)

            dice_total += compute_dice(outputs, masks) * images.size(0)
            iou_total += compute_iou(outputs, masks) * images.size(0)

            if save_pred:
                images_shown=0
                pred = torch.sigmoid(outputs)
                pred = (pred > 0.5).float()
                for i in range(images.shape[0]):
                    image = unnormalize(images[i].cpu())
                    image = image.permute(1, 2, 0).numpy()
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
                    plt.show()
                    plt.savefig(os.path.join(save_dir, f'pred_b{batch_idx}_{i}.png'))
                    plt.close()
                    images_shown += 1
                    if images_shown >= NUM_SAMPLES:
                        break

    avg_loss = val_loss / len(val_loader.dataset)
    avg_dice = dice_total / len(val_loader.dataset)
    avg_iou = iou_total / len(val_loader.dataset)

    print(f"Total Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")
    return avg_dice
    

def test(model, ckpt_path, test_loader, criterion, device, save_pred=False, save_dir='/data1/yoojinoh/codes/mlops/polyp-seg/output'):
    print("Loading model from", ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)
    print(f'Evaluating model on test set...')
    validate(model, test_loader, criterion, device, save_pred=save_pred, save_dir=save_dir)


def main(args):
    config = load_yaml(args.config)
    data_config = config['data']
    train_config = config['train']
    default_config = config['default'] 

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    batch_size = default_config['batch_size'] 
    num_workers = default_config['num_workers']
    lr = train_config['lr']
    epochs = train_config['epochs'] 
    print(f"Batch size: {batch_size}, Learning rate: {lr}, Epochs: {epochs}, Workers: {num_workers}")

    # Load model
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # binary segmentation
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = bce_dice_loss

    data_loaders = get_data_loader(
                            batch_size=batch_size, 
                            metadata_root = os.path.join(data_config['root'], 'metadata'), 
                            data_roots= data_config['root'],
                            workers=num_workers,
                            mean=default_config['mean'],
                            std=default_config['std'])

    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']

    if args.mode == 'train':
        model, train_losses, bce_losses, dice_losses = \
            train(epochs, train_loader, val_loader, model, optimizer, criterion, device)
        plot_losses(train_losses, bce_losses, dice_losses) 
        

    else:
        test(model, 
            args.ckpt_path, 
            test_loader, 
            criterion,
            device,
            save_pred=True,
            save_dir=os.path.join(data_config['save_dir'],'test'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate the model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--ckpt_path', type=str, default='/data1/yoojinoh/codes/mlops/ckpt/deeplabv3_epoch20.pth', help='Path to the model checkpoint')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode to run the script in (train/test)')
    args = parser.parse_args()

    main(args)