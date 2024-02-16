import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torchvision
from scipy.spatial import distance
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report  # Import confusion_matrix and classification_report

# Set up transformations for the dataset
transform = transforms.Compose([transforms.ToTensor()])

# Load CIFAR-10 train and test datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=1)

# CIFAR-10 class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Iterate over training data loader to fetch a batch of images and labels
for images, labels in trainloader:
    # Show images and print labels
    imshow(torchvision.utils.make_grid(images[:6]))
    print(' '.join('%5s' % classes[labels[j]] for j in range(6)))
    break  # Break after the first iteration to show only one batch

# Iterate over test data loader to fetch a batch of test images and labels
for test_images, test_labels in testloader:
    break  # Break after the first iteration to show only one batch

def euclidean_distance(p1, p2):
    d = np.linalg.norm(p1 - p2)
    return d

class KNeighborsClassifier:
    def __init__(self, distance_metric=0, n_neighbors=None):
        self.k = n_neighbors

    def fit(self, x, y):
        self.X_train = x
        self.Y_train = y

    def predict(self, x_test):
        predicted_labels = [self.nearest_neighbor(img) for img in x_test]
        np_predicted = np.array(predicted_labels)
        return np_predicted

    def nearest_neighbor(self, img):
        distance_matrix = [euclidean_distance(img, x_train) for x_train in self.X_train]
        closest_K = np.argsort(distance_matrix)[:self.k]
        closest_K_labels = [self.Y_train[i] for i in closest_K]
        majority_label = Counter(closest_K_labels).most_common(1)
        return majority_label[0][0]

def error_rate(y_predict, y_true):
    errorRate = np.mean(y_predict != y_true)
    return errorRate

# Convert training and test images to numpy arrays
x_train = np.array(images)
y_train = np.array(labels)
x_test = np.array(test_images)
y_test = np.array(test_labels)

MIN_K = 1
MAX_K = 15

def find_best_k(min_k=1, max_k=15):
    k_error = []
    for i in range(min_k, max_k):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        k_error.append(error_rate(pred_i, y_test))
    return k_error.index(min(k_error)) + 1, k_error

best_k, error_rate = find_best_k(min_k=MIN_K, max_k=MAX_K)

# Plot error rate vs. K
plt.figure(figsize=(10,6))
plt.plot(range(MIN_K, MAX_K), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error rate vs K')
plt.xlabel('K')
plt.ylabel('Error rate')

# Retrain with the best K value
knn = KNeighborsClassifier(best_k)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)

# Display confusion matrix and classification report
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))
