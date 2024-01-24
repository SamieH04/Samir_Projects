import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Assuming input image size is (height, length)
image_height = 200
image_length = 200
batch_size = 32

# Define the data transformations
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.Resize((image_height, image_length)),
    transforms.ToTensor(),
    transforms.Normalize((0.4, 0.4, 0.4), (0.6, 0.6, 0.6)),
])

def load_dataset(root_path, transform):
    dataset = datasets.ImageFolder(root=root_path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, loader

def extract_features_labels(loader):
    features = []
    labels = []

    for images, lbls in loader:
        flat_images = images.view(images.size(0), -1).numpy()
        features.append(flat_images)
        labels.append(lbls.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels

def standardize_features(train_feat, val_feat, test_feat):
    scaler = StandardScaler()
    train_feat_std = scaler.fit_transform(train_feat)
    val_feat_std = scaler.transform(val_feat)
    test_feat_std = scaler.transform(test_feat)

    return train_feat_std, val_feat_std, test_feat_std

# Load datasets
train_dataset, train_loader = load_dataset("/home/asfa/Documents/Dataset/train", transform)
validate_dataset, validate_loader = load_dataset("/home/asfa/Documents/Dataset/val", transform)
test_dataset, test_loader = load_dataset("/home/asfa/Documents/Dataset/test", transform)

# Extract features and labels
train_features, train_labels = extract_features_labels(train_loader)
val_features, val_labels = extract_features_labels(validate_loader)
test_features, test_labels = extract_features_labels(test_loader)

# Standardize features
train_features_std, val_features_std, test_features_std = standardize_features(train_features, val_features, test_features)

# Perform PCA
pca = PCA(n_components=100)
train_features_pca = pca.fit_transform(train_features_std)
val_features_pca = pca.transform(val_features_std)
test_features_pca = pca.transform(test_features_std)

# Train k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(train_features_pca, train_labels)

# Evaluate performance on validation set
val_predictions = knn_classifier.predict(val_features_pca)
val_accuracy = accuracy_score(val_labels, val_predictions)
classification_rep = classification_report(val_labels, val_predictions)
confusion_mat = confusion_matrix(val_labels, val_predictions)

# Print results for the validation set
print("On the Validation set:")
print("Validation Accuracy (k-NN with PCA): {:.3f}".format(val_accuracy))
print("Classification Report (k-NN with PCA):\n", classification_rep)
print("Confusion Matrix (k-NN with PCA):\n", confusion_mat)

# Evaluate performance on test set
test_predictions = knn_classifier.predict(test_features_pca)
test_accuracy = accuracy_score(test_labels, test_predictions)
test_classification_rep = classification_report(test_labels, test_predictions)
test_confusion_mat = confusion_matrix(test_labels, test_predictions)

# Print results for the test set
print("On the Test set:")
print("Test Accuracy (k-NN with PCA): {:.3f}".format(test_accuracy))
print("Test Classification Report (k-NN with PCA):\n", test_classification_rep)
print("Test Confusion Matrix (k-NN with PCA):\n", test_confusion_mat)
