import torch
import torch.nn as nn
from  torch.utils.data import DataLoader
import torchvision
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Define a CNN model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
net.to(device)

# Define a Loss function and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

    

# Save Model
PATH = 'ClassifierTraining/cifar_net.pt'
torch.save(net.state_dict(), PATH)






# Accuracy Calculation
def accuracy(testloader):
    # net = model   
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            outputs = outputs.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)*100)


def main():

    # Loading and Normalize CIFAR10

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # Mean and Standard Deviation
    )

    batch_size = 8
    trainset = datasets.CIFAR10(root = 'data/Cifar', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle = True, num_workers=2)

    testset = datasets.CIFAR10(root='data/Cifar', train = False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat','deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    

    # get some random training images
    # dataiterTrain = iter(trainloader)
    # imagesTrain, labelsTrain = next(dataiterTrain)

    # dataitertest = iter(testloader)
    # imagesTest, labelsTest = next(dataitertest)

    # show images
    # imshow(torchvision.utils.make_grid(imagesTrain))
    # imshow(torchvision.utils.make_grid(imagesTest))

    # print labels
    # print(' '.join(f'{classes[labelsTrain[j]]:5s}' for j in range(batch_size)))
    # print(' '.join(f'{classes[labelsTest[j]]:5s}' for j in range(batch_size)))
    
    
    # Train the Network
    epochs = 25
    for epoch in range(epochs):
        running_loss = 0.0
        for i,data in enumerate(trainloader, 0):
            # get the inputs; data is the list of [inputs, labels]
            inputs , labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print Statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch +1}, {i+1:7d}] loss: {running_loss/2000:.3f}')
                running_loss = 0.0
            
    print('Finished Training')
    
    # Test the network on the test data

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    
    # Loading Saved Model
    
    net.load_state_dict(torch.load(PATH))
    
    outputs = net(images.to(device))
    
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))   

    print("Accuracy: .2f ",accuracy(testloader))    




    
    
if __name__ == "__main__":
    main()