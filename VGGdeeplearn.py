# Import necessary libraries
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import matplotlib.pyplot as plt

# Hyperparameters
image_height = 200
image_length = 200
batch_size = 32
epochs = 10
lr = 0.001
dropout = 0.1

# Dataset paths
train_dataset_path = "Directory"
validate_dataset_path = "Directory"
test_dataset_path = "Directory"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available

# Define the data transformations
transform = transforms.Compose([
    transforms.RandomRotation(15),  # Randomly rotate images by up to 15 degrees
    transforms.Resize((image_height, image_length)),  # Resize images to a specified size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.4, 0.4, 0.4), (0.6, 0.6, 0.6)),  # Normalize image channels
])
    
def load_dataset(root_path, transform):
    dataset = datasets.ImageFolder(root=root_path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, loader

# Load the training dataset
train_dataset, train_loader = load_dataset(train_dataset_path, transform)

# Load the validation dataset
validate_dataset, validate_loader = load_dataset(validate_dataset_path, transform)

# Load the test dataset
test_dataset, test_loader = load_dataset(test_dataset_path, transform)

# Load the VGG16 model architecture
model = models.vgg16(pretrained=False)

# Allow parameters to be updated during training
for parameter in model.parameters():
    parameter.requires_grad = True

# Modify the classifier part of the VGG16 model
classifier = nn.Sequential(OrderedDict([
    ('fcl', nn.Linear(model.classifier[0].in_features, 500)),
    ('relu', nn.ReLU()),
    ('drop', nn.Dropout(p=dropout)),
    ('fc2', nn.Linear(500, 10)),
    ('output', nn.LogSoftmax(dim=1))
]))
model.classifier = classifier

# Calculate and print the number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())

# Lists to store training and validation loss and accuracy
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Function to perform validation during training
def validation(model, loader, criterion):
    loss = 0
    accuracy = 0

    model.eval()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            output = model(images)
            loss += criterion(output, labels).item()

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += equality.type(torch.FloatTensor).mean().item()

    model.train()

    return loss, accuracy

# Function to test the model on the test dataset
def test(model, loader, criterion):
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to store test loss and accuracy
    test_loss = 0
    test_accuracy = 0

    # Iterate through the test dataset
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        with torch.no_grad():
            output = model(images)
            test_loss += criterion(output, labels).item()

            # Calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            test_accuracy += equality.type(torch.FloatTensor).mean().item()

    # Calculate average test loss and accuracy
    avg_test_loss = test_loss / len(test_loader)
    avg_test_accuracy = test_accuracy / len(test_loader)

    # Print the results
    print(f"Test Loss: {avg_test_loss:.3f}")
    print(f"Test Accuracy: {avg_test_accuracy:.3f}")

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)

# Function to train the classifier
def train_classifier():
    # Move the model to the specified device
    model.to(device)
    
    for e in range(epochs):
        start_time = time.time()  # Record the start time for the epoch
        running_loss = 0 # Reset loss for each epoch
        running_accuracy = 0 # Reset accuracy for each epoch
        steps = 0  # Reset steps for each epoch

        # Iterate through the training dataset
        for images, labels in iter(train_loader):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            running_accuracy += equality.type(torch.FloatTensor).mean().item()

        # Calculate validation loss and accuracy
        val_loss, val_accuracy = validation(model, validate_loader, criterion)

        # Append metrics for plotting
        train_losses.append(running_loss / steps)
        val_losses.append(val_loss / len(validate_loader))
        train_accuracies.append(running_accuracy / steps)
        val_accuracies.append(val_accuracy / len(validate_loader))

        print("Epoch: {}/{}.. ".format(e + 1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / steps),
              "Validation Loss: {:.3f}.. ".format(val_loss / len(validate_loader)),
              "Training Accuracy: {:.3f}.. ".format(running_accuracy / steps),
              "Validation Accuracy: {:.3f}".format(val_accuracy / len(validate_loader)),
              f"Time: {time.time() - start_time:.2f} seconds")

#Code above is built on from this Youtube Video: https://www.youtube.com/watch?v=zFA8Cm13Xmk&t=0s
    # Plotting after training is complete
    f, ax = plt.subplots(2, 1, figsize=(5, 5))
    ax[0].plot(train_losses, color='b', label='Training Loss')
    ax[0].plot(val_losses, color='r', label='Validation Loss')
    ax[0].legend(loc="upper right")

    ax[1].plot(train_accuracies, color='b', label='Training Accuracy')
    ax[1].plot(val_accuracies, color='r', label='Validation Accuracy')
    ax[1].legend(loc="lower right")

    plt.show()

# Function to show sample images from the dataset
def show_sample_images(loader, title, dataset):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images = images.numpy()

    # Map numerical labels to class names
    class_names = [key for key, value in dataset.class_to_idx.items()]
    class_labels = [class_names[label] for label in labels.numpy()]

    # Display images along with their labels
    fig, axes = plt.subplots(figsize=(15, 3), ncols=8)
    for i in range(8):
        axes[i].imshow(images[i].transpose((1, 2, 0)))
        axes[i].set_title(f'Label: {class_labels[i]}')
        axes[i].axis('off')
    fig.suptitle(title, fontsize=16)
    plt.show()

# Show sample images from the training dataset
show_sample_images(train_loader, title='Sample Images from Training Dataset', dataset=train_dataset)

# Show sample images from the validation dataset
show_sample_images(validate_loader, title='Sample Images from Validation Dataset', dataset=validate_dataset)

# Show sample images from the test dataset
show_sample_images(test_loader, title='Sample Images from Test Dataset', dataset=test_dataset)

# Train the classifier
print(f'Total number of parameters in the model: {total_params}')  # Print again for reference
print(f"Training on device: {device}")
train_classifier()

# Test the trained model on an unseen dataset
test(model, test_loader, criterion)
