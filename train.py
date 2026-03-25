import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS_P1  = 17
EPOCHS_P2  = 15


# TRAIN FUNCTION

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct = 0, 0

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


# MAIN FUNCTION

def main():
    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # PATHS
    base_path = r"C:\Users\kriti\Downloads\PlantVillage"
    train_dir = os.path.join(base_path, "train")
    val_dir   = os.path.join(base_path, "val")

    # TRANSFORMS
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # DATASET
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)

    # Save labels
    with open("labels.txt", "w") as f:
        for c in class_names:
            f.write(c + "\n")

    # MODEL
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze base (Phase 1)
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    model = model.to(device)

    # LOSS + OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)

    scaler = torch.amp.GradScaler("cuda")

    # PHASE 1
  
    print("\n========== PHASE 1 ==========")

    best_acc = 0
    patience_counter = 0

    for epoch in range(EPOCHS_P1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_phase1.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 4:
            print("Early stopping (Phase 1)")
            break

    # PHASE 2 (FINE-TUNE)
    
    print("\n========== PHASE 2 ==========")

    for param in model.features.parameters():
        param.requires_grad = False

    for param in list(model.features.parameters())[-80:]:
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    best_acc = 0
    patience_counter = 0

    for epoch in range(EPOCHS_P2):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"[FT] Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_phase2.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 5:
            print("Early stopping (Phase 2)")
            break

    # SAVE FINAL MODEL
    
    torch.save(model.state_dict(), "model_best.pth")
    print("✅ Final model saved")

    
    # FINAL EVAL
    
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Final Val Accuracy: {val_acc*100:.2f}%")



# ENTRY POINT (FIXES YOUR ERROR)

if __name__ == "__main__":
    main()