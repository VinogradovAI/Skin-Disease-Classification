import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import warnings
from timm import create_model
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import pandas as pd

# Ignore warnings
warnings.filterwarnings("ignore")

# Paths
PROCESSED_PATH = Path("D:/ml_school/Skillfactory/CV/FP/processed_data")
BATCH_SIZE = 32
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Function to load dataset
def get_dataloaders():
    train_dataset = datasets.ImageFolder(root=PROCESSED_PATH / "train/img", transform=train_transforms)
    valid_dataset = datasets.ImageFolder(root=PROCESSED_PATH / "valid/img", transform=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, valid_loader, len(train_dataset.classes)


# Training function
def train_model(model, train_loader, valid_loader, num_classes, freeze_epochs=5):
    if hasattr(model, 'fc'):
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    elif hasattr(model, 'classifier'):
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    elif hasattr(model, 'head'):  # Fix for Swin Transformer
        model.head = nn.Linear(model.head.in_features, num_classes)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_f1_scores = []
    epoch_times = []

    print(f"Training {model.__class__.__name__} on {DEVICE}...")
    best_val_acc = 0.0
    early_stop_counter = 0

    # Для Confusion Matrix
    all_labels, all_preds = [], []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)

            if len(outputs.shape) == 4:
                outputs = outputs.mean(dim=[2, 3])

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_losses.append(running_loss)
        train_accuracies.append(train_acc)
        scheduler.step()

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                if len(outputs.shape) == 4:
                    outputs = outputs.mean(dim=[2, 3])
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Keep predictions for Confusion Matrix
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Calculate F1
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_f1_scores.append(val_f1)

        epoch_times.append(time.time() - start_time)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}, Time: {epoch_times[-1]:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter > 3:
                print("Early stopping triggered.")
                break

    # Plots after epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model.__class__.__name__} Loss per Epoch')
    plt.legend()
    plt.savefig(f"metrics/{model.__class__.__name__}_loss_plot.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model.__class__.__name__} Accuracy per Epoch')
    plt.legend()
    plt.savefig(f"metrics/{model.__class__.__name__}_accuracy_plot.png")
    plt.show()

    # Confusion Matrix after training
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=range(num_classes), columns=range(num_classes))

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model.__class__.__name__}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"metrics/{model.__class__.__name__}_confusion_matrix.png")
    plt.show()

    print(f"Average epoch time: {sum(epoch_times) / len(epoch_times):.2f}s")

    # Save model to directory 'models'
    torch.save(model.state_dict(), f"models/{model.__class__.__name__}.pth")

    # Save metrics to directory 'metrics'
    with open(f"metrics/{model.__class__.__name__}.txt", "w") as f:
        f.write(f"Train Loss: {train_losses}\n")
        f.write(f"Validation Loss: {val_losses}\n")
        f.write(f"Train Accuracy: {train_accuracies}\n")
        f.write(f"Validation Accuracy: {val_accuracies}\n")
        f.write(f"Validation F1: {val_f1_scores}\n")
        f.write(f"Average epoch time: {sum(epoch_times) / len(epoch_times):.2f}s\n")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train_loader, valid_loader, num_classes = get_dataloaders()

    models_dict = {
        "resnet50": models.resnet50(pretrained=True).to(DEVICE),
        "efficientnet_b4": models.efficientnet_b4(pretrained=True).to(DEVICE),
        "swin_transformer": create_model("swin_tiny_patch4_window7_224", pretrained=True).to(DEVICE)
    }

    for model_name, model in models_dict.items():
        train_model(model, train_loader, valid_loader, num_classes)