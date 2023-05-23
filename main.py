import os

from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

# Local modules
from models.alexnet import AlexNet


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = args.batch_size
        self.epochs = args.epochs

        self.get_dataset(args.batch_size, args.data_root)

        self.model = AlexNet().to("cuda").eval()
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters())

    def get_dataset(self, batch_size, data_root):
        tr = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5)),
                transforms.Resize((224, 224)),
            ]
        )
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=tr
        )
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=tr
        )
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

    def train_step(self, data):
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.loss_func(outputs, labels)
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def validation_step(self, data):
        images, labels = data
        images, labels = images.to(self.device), labels.to(self.device)

        outputs = self.model(images)

        _, predicted = torch.max(outputs.data, 1)

        return predicted, labels

    def train_loop(self):
        os.makedirs("checkpoints", exist_ok=True)
        for epoch in range(self.epochs):
            # Train loop
            self.model.train()
            for _, data in enumerate(tqdm(self.train_dataloader)):
                self.train_step(data)

            # Validation loop
            with torch.no_grad():
                correct = 0
                total = 0
                self.model.eval()
                for _, data in enumerate(tqdm(self.test_dataloader)):
                    predicted, labels = self.validation_step(data)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = correct / total
                print(f"Epoch {epoch}: accuracy: {accuracy}")
                # Log result
                with open("log.txt", "a") as stream:
                    stream.write(f"Epoch {epoch}: accuracy: {accuracy}\n")

            # Save checkpoints
            torch.save(self.model.state_dict(), f"checkpoints/epoch{epoch}.pt")


def main(args):
    trainer = Trainer(args)
    trainer.train_loop()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-d", "--data_root", type=str, default="./data")
    parser.add_argument("-e", "--epochs", type=int, default=90)

    args = parser.parse_args()
    main(args)
