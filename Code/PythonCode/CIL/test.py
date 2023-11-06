from utils.data import KDD99, iCIFAR100
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(filename)s] => %(message)s",
                    handlers=[logging.StreamHandler()]
                    )
kdd99 = KDD99()
kdd99.download_data()
print(len(kdd99.test_targets))
cifar100 = iCIFAR100()
cifar100.download_data()

# iCIFAR100().download_data()


class DummyDataset(Dataset):
    def __init__(self, images, labels, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        # self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # if self.use_path:
        #     image = self.trsf(pil_loader(self.images[idx]))
        # else:
        #     image = self.trsf(Image.fromarray(self.images[idx]))
        image = self.images[idx]
        label = self.labels[idx]
        return idx, image, label


dataset = DummyDataset(kdd99.train_data, kdd99.train_targets)
val_dataset = DummyDataset(kdd99.test_data, kdd99.test_targets)


class IntrusionDetectionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(IntrusionDetectionNet, self).__init__()
        self.input_dim = input_dim

        # Define the network layers.
        self.fc1 = nn.Linear(in_features=self.input_dim, out_features=128)
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: Tensor):
        # Pass the input data through the network layers.
        x = self.layers(x)
        plt.imshow(x.detach().numpy())
        plt.show()

        # x = self.fc1(x)
        return x


model = IntrusionDetectionNet(121, 23)

# Define the loss function and optimizer.
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

prog_bar = tqdm(range(100))
# Train the model for 10 epochs
for epoch in prog_bar:
    # Get the training data
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    # Train the model
    for batch_idx, (_, data, target) in enumerate(train_loader):
        model.train()
        # Forward pass
        output = model(data)
        # Calculate the loss
        loss = F.cross_entropy(output, target)
        logging.info(f"Loss is: {loss}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break

# Get the validation data
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Evaluate the model
correct = 0
total = 0
for batch_idx, (_, data, target) in enumerate(val_loader):
    # Forward pass
    model.eval()
    output = model(data)

    # Calculate the predictions
    _, predicted = torch.max(output.data, 1)

    # Update the correct and total counters
    total += target.size(0)
    correct += (predicted == target).sum().item()

# Calculate the accuracy
accuracy = 100 * correct / total

print('Validation accuracy:', accuracy)
