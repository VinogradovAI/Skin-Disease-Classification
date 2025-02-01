import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from pathlib import Path

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
PROCESSED_PATH = Path("D:/ml_school/Skillfactory/CV/FP/processed_data")

# Define hyperparameter search space
space = [
    Real(1e-5, 1e-3, name='lr'),
    Real(1e-5, 1e-2, name='weight_decay'),
    Real(0.2, 0.5, name='dropout'),
    Categorical([True, False], name='use_mixup'),
    Categorical([True, False], name='fine_tune_all')
]


# Load datasets
def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=PROCESSED_PATH / "train/img", transform=transform)
    valid_dataset = datasets.ImageFolder(root=PROCESSED_PATH / "valid/img", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, len(train_dataset.classes)


# Training function
def train_model(model, train_loader, valid_loader, lr, weight_decay, dropout, use_mixup, fine_tune_all, epochs=5):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Freeze or unfreeze layers based on fine-tuning strategy
    for param in model.parameters():
        param.requires_grad = fine_tune_all

    # Ensure classification head is always trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate model on validation set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    return -val_acc  # Minimize negative accuracy for optimization


# Objective function for optimization
@use_named_args(space)
def objective(**params):
    batch_size = 32  # Fixed batch size for now
    train_loader, valid_loader, num_classes = get_dataloaders(batch_size)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(params['dropout']),
        nn.Linear(512, num_classes)
    )

    return train_model(model, train_loader, valid_loader, **params)


# Run Bayesian Optimization
def tune_hyperparameters():
    print("Starting hyperparameter tuning...")
    result = gp_minimize(objective, space, n_calls=10, random_state=42)
    print("Best hyperparameters found:")
    print(result.x)
    return result


if __name__ == "__main__":
    best_result = tune_hyperparameters()
