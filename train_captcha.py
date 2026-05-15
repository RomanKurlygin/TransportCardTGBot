from pathlib import Path
import random

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms


DATASET_DIR = Path("dataset/captcha/labeled")
MODEL_PATH = Path("models/captcha_cnn.pth")
MODEL_PATH.parent.mkdir(exist_ok=True)


class CaptchaDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

        if len(self.files) == 0:
            raise ValueError("Список captcha пустой")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_text = path.stem[:4]

        if not label_text.isdigit() or len(label_text) != 4:
            raise ValueError(f"Неверное имя файла: {path.name}")

        label = torch.tensor([int(c) for c in label_text], dtype=torch.long)

        return image, label


class CaptchaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 20, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.head1 = nn.Linear(256, 10)
        self.head2 = nn.Linear(256, 10)
        self.head3 = nn.Linear(256, 10)
        self.head4 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        d1 = self.head1(x)
        d2 = self.head2(x)
        d3 = self.head3(x)
        d4 = self.head4(x)

        return d1, d2, d3, d4


def calculate_loss(outputs, labels, criterion):
    d1, d2, d3, d4 = outputs

    loss = (
        criterion(d1, labels[:, 0]) +
        criterion(d2, labels[:, 1]) +
        criterion(d3, labels[:, 2]) +
        criterion(d4, labels[:, 3])
    )

    return loss


def calculate_accuracy(outputs, labels):
    d1, d2, d3, d4 = outputs

    p1 = d1.argmax(dim=1)
    p2 = d2.argmax(dim=1)
    p3 = d3.argmax(dim=1)
    p4 = d4.argmax(dim=1)

    full_correct = (
        (p1 == labels[:, 0]) &
        (p2 == labels[:, 1]) &
        (p3 == labels[:, 2]) &
        (p4 == labels[:, 3])
    ).sum().item()

    symbol_correct = (
        (p1 == labels[:, 0]).sum().item() +
        (p2 == labels[:, 1]).sum().item() +
        (p3 == labels[:, 2]).sum().item() +
        (p4 == labels[:, 3]).sum().item()
    )

    total_samples = labels.size(0)
    total_symbols = labels.size(0) * 4

    return full_correct, symbol_correct, total_samples, total_symbols


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_full_correct = 0
    total_symbol_correct = 0
    total_samples = 0
    total_symbols = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = calculate_loss(outputs, labels, criterion)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        full_correct, symbol_correct, samples, symbols = calculate_accuracy(
            outputs, labels
        )

        total_full_correct += full_correct
        total_symbol_correct += symbol_correct
        total_samples += samples
        total_symbols += symbols

    avg_loss = total_loss / len(loader)
    full_acc = total_full_correct / total_samples
    symbol_acc = total_symbol_correct / total_symbols

    return avg_loss, full_acc, symbol_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_full_correct = 0
    total_symbol_correct = 0
    total_samples = 0
    total_symbols = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = calculate_loss(outputs, labels, criterion)

        total_loss += loss.item()

        full_correct, symbol_correct, samples, symbols = calculate_accuracy(
            outputs, labels
        )

        total_full_correct += full_correct
        total_symbol_correct += symbol_correct
        total_samples += samples
        total_symbols += symbols

    avg_loss = total_loss / len(loader)
    full_acc = total_full_correct / total_samples
    symbol_acc = total_symbol_correct / total_symbols

    return avg_loss, full_acc, symbol_acc


def main():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    real_files = sorted(list(Path("dataset/captcha/labeled").glob("*.png")))
    synthetic_files = sorted(list(Path("dataset/captcha/synthetic").glob("*.png")))

    if len(real_files) == 0 and len(synthetic_files) == 0:
        raise ValueError(
            "Нет captcha для обучения. "
            "Проверь папки dataset/captcha/labeled и dataset/captcha/synthetic"
        )

    print(f"Реальных captcha: {len(real_files)}")
    print(f"Синтетических captcha: {len(synthetic_files)}")

    # Перемешиваем реальные captcha
    random.shuffle(real_files)
    random.shuffle(synthetic_files)

    # Validation делаем только на реальных captcha
    val_size = max(1, int(0.2 * len(real_files)))

    val_files = real_files[:val_size]
    train_real_files = real_files[val_size:]

    # Train = реальные captcha без validation + synthetic captcha
    train_files = train_real_files + synthetic_files
    random.shuffle(train_files)

    print(f"Train real captcha: {len(train_real_files)}")
    print(f"Train synthetic captcha: {len(synthetic_files)}")
    print(f"Train total captcha: {len(train_files)}")
    print(f"Val real captcha: {len(val_files)}")

    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((50, 160)),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((50, 160)),
        transforms.ToTensor()
    ])

    train_dataset = CaptchaDataset(train_files, transform=train_transform)
    val_dataset = CaptchaDataset(val_files, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )

    model = CaptchaCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005
    )

    num_epochs = 50

    best_val_symbol_acc = 0.0
    best_val_full_acc = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_full_acc, train_symbol_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_full_acc, val_symbol_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_full_acc={train_full_acc:.4f} | "
            f"train_symbol_acc={train_symbol_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_full_acc={val_full_acc:.4f} | "
            f"val_symbol_acc={val_symbol_acc:.4f}"
        )

        if val_symbol_acc > best_val_symbol_acc:
            best_val_symbol_acc = val_symbol_acc
            best_val_full_acc = val_full_acc
            best_epoch = epoch

            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Модель сохранена: {MODEL_PATH}")

    print("Обучение завершено.")
    print(f"Лучшая эпоха: {best_epoch}")
    print(f"Лучшая val_symbol_acc: {best_val_symbol_acc:.4f}")
    print(f"Лучшая val_full_acc: {best_val_full_acc:.4f}")
    print(f"Файл модели: {MODEL_PATH}")


if __name__ == "__main__":
    main()