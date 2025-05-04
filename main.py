from torchvision.models.segmentation import deeplabv3_resnet101
import torch.nn as nn
from dataset import get_data_loader 
import torch 
import os 
import argparse
import logging
from datetime import datetime
import shutil 

from utils import load_yaml, config_to_tag, visualize_predictions
from utils import plot_losses, compute_dice, compute_iou, compute_accuracy, compute_precision, compute_recall

NUM_SAMPLES = 5

# Loss
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def bce_dice_loss(pred, target, w1=1.0, w2=1.0):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return {
        'loss': w1*bce + w2*dice,
        'bce_loss': bce,
        'dice_loss': dice
    }


def train(num_epochs, train_loader, val_loader, model, optimizer, scheduler, criterion, device, ckpt_save_dir='/data1/yoojinoh/codes/mlops/ckpt',):
    logging.info("Training the model...")
    train_losses = [] 
    bce_losses = []
    dice_losses = []
    best_dice = 0.0
    os.makedirs(ckpt_save_dir, exist_ok=True)

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
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f} (BCE: {avg_bce_loss:.4f}, Dice: {avg_dice_loss:.4f})")

        # Validation
        val_dice = validate(model, 
                val_loader, 
                criterion, 
                device, 
                save_pred=False)
        
        scheduler.step(val_dice) # ReduceLROnPlateau; When val_dice does not improve, reduce the learning rate

        if best_dice < val_dice:
            best_dice = val_dice
            best_model_path = os.path.join(ckpt_save_dir, 'deeplabv3_best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved to {best_model_path}")

        if epoch == (num_epochs - 1):
            os.makedirs(ckpt_save_dir, exist_ok=True)
            save_name = f'deeplabv3_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), os.path.join(ckpt_save_dir, save_name))
            logging.info(f"Last Model saved to {os.path.join(ckpt_save_dir, save_name)}")

    logging.info('Training complete!')
    return model, best_model_path, train_losses, bce_losses, dice_losses

def validate(model, val_loader, criterion, device, save_pred=False, save_dir='/data1/yoojinoh/codes/mlops/polyp-seg/output'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    val_loss = 0.0
    dice_total = 0.0
    iou_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    accuracy_total = 0.0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)['loss']
            val_loss += loss.item() * images.size(0)

            dice_total += compute_dice(outputs, masks) * images.size(0)
            iou_total += compute_iou(outputs, masks) * images.size(0)
            precision_total += compute_precision(outputs, masks) * images.size(0)
            recall_total += compute_recall(outputs, masks) * images.size(0)
            accuracy_total += compute_accuracy(outputs, masks) * images.size(0)

            if save_pred:
                visualize_predictions(images, masks, outputs, batch_idx, save_dir, num_samples=NUM_SAMPLES)

    avg_loss = val_loss / len(val_loader.dataset)
    avg_dice = dice_total / len(val_loader.dataset)
    avg_iou = iou_total / len(val_loader.dataset)

    avg_precision = precision_total / len(val_loader.dataset)
    avg_recall = recall_total / len(val_loader.dataset)
    avg_accuracy = accuracy_total / len(val_loader.dataset)

    logging.info(f"Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, "
             f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, Acc: {avg_accuracy:.4f}")

    return avg_dice
    

def test(model, ckpt_path, test_loader, criterion, device, save_pred=False, save_dir='/data1/yoojinoh/codes/mlops/polyp-seg/output'):
    logging.info("Loading model from", ckpt_path)

    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)

    logging.info(f'Evaluating model on test set...')
    validate(model, test_loader, criterion, device, save_pred=save_pred, save_dir=save_dir)


def main(args):
    config = load_yaml(args.config)

    data_config = config['data']
    train_config = config['train']
    default_config = config['default'] 

    tag = config_to_tag(config)
    os.makedirs(os.path.join(data_config['save_dir'], tag), exist_ok=True)
    data_config['save_dir'] = os.path.join(data_config['save_dir'], tag) 
    shutil.copy(args.config, os.path.join(data_config['save_dir'], 'config.yaml'))
    
    log_file = os.path.join(data_config['save_dir'], f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  
        ]
    )
    logging.info(f"Experiment: {tag}")

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    batch_size = default_config['batch_size'] 
    num_workers = default_config['num_workers']
    lr = train_config['lr']
    epochs = train_config['epochs'] 
    logging.info(f"Batch size: {batch_size}, Learning rate: {lr}, Epochs: {epochs}, Workers: {num_workers}")
    
    # Load model
    logging.info("Loading model...")
    model = deeplabv3_resnet101(pretrained=True)
    print(model)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # binary segmentation-> Conv(256, 21) to Conv(256, 1)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
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

    if args.mode in ('train', 'all'):
        model, best_model_path, train_losses, bce_losses, dice_losses = \
            train(epochs, train_loader, val_loader, model, optimizer, scheduler, criterion, device, ckpt_save_dir=os.path.join(data_config['save_dir'], 'ckpt'))
        plot_losses(train_losses, bce_losses, dice_losses, save_root=os.path.join(data_config['save_dir'], 'output')) 
        

    if args.mode in ('test', 'all'):
        if args.mode == 'all':
            args.ckpt_path = best_model_path
        test(model, 
            args.ckpt_path, 
            test_loader, 
            criterion,
            device,
            save_pred=True,
            save_dir=os.path.join(data_config['save_dir'],'output/test'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate the model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--ckpt_path', type=str, default='/data1/yoojinoh/codes/mlops/polyp-seg/bs32_lr1e-04_ep50/ckpt/deeplabv3_best_model.pth', help='Path to the model checkpoint')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'all'], help='Mode to run the script in (train/test)')
    args = parser.parse_args()

    main(args)