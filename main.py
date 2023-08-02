# %%
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 4

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# %%
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


images, labels = next(iter(trainloader))

imshow(torchvision.utils.make_grid(images))
print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from toolz import pipe


def flatten_1(t: torch.Tensor):
    return torch.flatten(t, 1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, inp):
        return pipe(
            inp,
            self.conv1,
            F.relu,
            self.pool,
            self.conv2,
            F.relu,
            self.pool,
            flatten_1,
            self.fc1,
            F.relu,
            self.fc2,
            F.relu,
            self.fc3,
        )


net = Net()
# %%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# %%
for epoch in range(2):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f"[{epoch}, {i:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")

# %%
PATH = "./cifar_net.pth"
torch.save(net.state_dict(), PATH)
# %%
images, labels = next(iter(testloader))

imshow(torchvision.utils.make_grid(images))
print("Ground Truth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))

# %%
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print("Predicted: ", " ".join(f"{classes[predicted[j]]:5s}" for j in range(4)))
# %%
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the network on the 10000 test images: {100 * correct // total} %")
# %%
from collections import defaultdict

correct_pred = defaultdict(int)
total_pred = defaultdict(int)

with torch.no_grad():
    for images, labels in testloader:
        outputs = net(images)
        _, predictions = torch.max(outputs.data, 1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

for name, correct_count in correct_pred.items():
    accuracy = 100 * correct_count / total_pred[name]
    print(f"Accuracy for class: {name:5s} is {accuracy:.1f} %")
# %%
