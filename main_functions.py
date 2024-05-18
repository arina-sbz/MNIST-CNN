import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix
import seaborn as sns

class NeuralNet(nn.Module):
    def __init__(self, layer_structure):
        """
        Initialize the neural network with the given layer structure

        Parameters:
            layer_structure: List of integers where each integer represents
                                    the number of neurons in that layer
        """
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()

        # Loop through the layer_structure to create each layer
        for i in range(len(layer_structure) - 1):
            self.layers.append(
                nn.Linear(layer_structure[i], layer_structure[i + 1]))

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network

        Parameters:
            x: Input tensor

        Returns:
            Output tensor
        """
        
        # Apply all layers except the last with ReLU activation
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.relu(x)

        # The last layer does not have ReLU if it's an output layer
        x = self.layers[-1](x)
        return x
    
class ConvNet(nn.Module):
    def __init__(self, layers_config):
        """
        Initialize the convolutional neural network with the given layer configuration

        Parameters:
            layers_config: List of dictionaries where each dictionary contains
                                  the type and structure of the layer
        """
        super(ConvNet, self).__init__()
        layers = []
        for layer in layers_config:
            type = layer['type']
            if 'structure' in layer:
                layer_params = layer['structure']
                layers.append(getattr(nn, type)(**layer_params))
            else:
                layers.append(getattr(nn, type)())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)    

def train_model(model, train_loader, test_loader, num_epochs, learning_rate, device, isCNN, optimizer_type, use_scheduler):
    """
    Train the model
    
    Parameters:
        model: The model to be trained
        train_loader: DataLoader for the training data
        test_loader: DataLoader for the testing data
        num_epochs: The number of epochs to train the model
        learning_rate: Learning rate for the optimizer
        device: Device to train the model on (CPU or GPU)
        isCNN: Boolean indicating if the model is a CNN (True) or not (False)
        optimizer_type: Type of optimizer to use
        use_scheduler: Boolean indicating if a learning rate scheduler should be used

    Returns:
        train_losses: List of training losses for each epoch
        train_accuracies: List of training accuracies for each epoch
        test_losses: List of testing losses for each epoch
        test_accuracies: List of testing accuracies for each epoch
    """
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    loss_function = nn.CrossEntropyLoss()
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
   
    if use_scheduler:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
    for epoch in range(num_epochs):
        batch_losses = []
        batch_correct_predictions = 0
        batch_total_predictions = 0
        for i, (images, labels) in enumerate(train_loader):
            # reshape the image
            if isCNN:
                images = images.reshape(-1, 1, 28, 28).to(device)
            else:
                images = images.reshape(-1, 28*28).to(device)
            true_labels = labels.to(device)

            # Forward pass
            predicted_labels = model(images)
            loss = loss_function(predicted_labels, true_labels)
            batch_losses.append(loss.item())
            
            _, prediction_indices = torch.max(predicted_labels, 1)
            _, true_labels_indices = torch.max(true_labels, 1)
            batch_correct_predictions += (prediction_indices ==
                                    true_labels_indices).sum().item()
            batch_total_predictions += true_labels.size(0)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(
                    f'For Epoch [{epoch+1}/{num_epochs}], the Loss is: {loss.item():.4f}')
                
        # Evaluate the model on the test data    
        epoch_test_loss, epoch_test_accuracy = test_model(model, test_loader, device, isCNN, False)
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_accuracy)
        
        if use_scheduler:
            scheduler.step()
        
        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)
        epoch_accuracy = 100 * (batch_correct_predictions / batch_total_predictions)
        train_accuracies.append(epoch_accuracy)   
         
    return train_losses, train_accuracies, test_losses, test_accuracies    
        
def test_model(model, test_loader, device, isCNN, isBestModel):
    """
    Test the model

    Parameters:
        model: The model to be tested
        test_loader: DataLoader for the testing data
        device: Device to test the model on (CPU or GPU)
        isCNN: Boolean indicating if the model is a CNN (True) or not (False)
        isBestModel: Boolean indicating if the model is considered the best model or not

    Returns:
        If isBestModel is True:
            true_labels_all: List of all true labels in the test set
            predicted_labels_all: List of all predicted labels in the test set
            misclassified_samples: List of misclassified samples with their true and predicted labels
        Otherwise:
            average_test_loss: Average loss over the test set
            accuracy: Accuracy of the model on the test set
    """    
    
    loss_function = nn.CrossEntropyLoss()
    correct_predictions = 0
    true_labels_all = []
    predicted_labels_all = []
    misclassified_samples = []
    isSeen = []
    test_losses = []
    n = len(test_loader.dataset)
    with torch.no_grad():
        for images, labels in test_loader:
            if isCNN:
                images = images.reshape(-1, 1, 28, 28).to(device)
            else:
                images = images.reshape(-1, 28*28).to(device)
            
            true_labels = labels.to(device)
            predicted_labels = model(images)
            
            loss = loss_function(predicted_labels, true_labels)
            test_losses.append(loss.item())
    
            _, prediction_indices = torch.max(predicted_labels, 1)
            # Convert one-hot encoded labels to class indices
            _, true_labels_indices = torch.max(true_labels, 1)
            correct_predictions += (prediction_indices ==
                                    true_labels_indices).sum().item()
            
            if isBestModel:   
                true_labels_all.extend(true_labels_indices.cpu().numpy())
                predicted_labels_all.extend(prediction_indices.cpu().numpy())
                misclassified_indices = (prediction_indices != true_labels_indices)
                            
                for i in range(len(images)):
                    if misclassified_indices[i]:  # Only consider misclassified cases
                        true_label = true_labels_indices[i].item()
                        if true_label not in isSeen:
                            image = images[i]
                            prediction = prediction_indices[i]
                            misclassified_samples.append((image, true_label, prediction.item()))
                            isSeen.append(true_label)             
                           
    accuracy = 100 * (correct_predictions / n)
    average_test_loss = np.mean(test_losses)
    
    print(f'The accuracy of this network on the test set is: {accuracy} %')
    if isBestModel:
        return true_labels_all, predicted_labels_all, misclassified_samples
    else:
        return average_test_loss, accuracy

def show_confusion_matrix(true_labels, predictions):
    """
    Show the confusion matrix for the true and predicted labels

    Parameters:
        true_labels: List or array of true labels
        predictions: List or array of predicted labels
    """
    matrix = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig("cm.png",dpi=300) 
    plt.show()
    
def show_misclassified_samples(misclassified_samples):
    """
    Show misclassified samples

    Parameters:
        misclassified_samples: List of tuples containing misclassified samples and their true and predicted labels
    """
    plt.figure(figsize=(10, 5))
    for i, (image, true_label, predicted_label) in enumerate(misclassified_samples):
        image = image.squeeze()
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.cpu().numpy(), cmap='gray')
        plt.title(f'True: {true_label}, Pred: {predicted_label}')
        plt.axis('off')
    plt.savefig("misclassified_samples.png",dpi=300) 
    plt.show()      

def training_curve_plot(title, subtitle,train_losses, test_losses, train_accuracy, test_accuracy):
    lg=13
    md=10
    sm=9
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=lg)
    plt.figtext(0.5, 0.90, subtitle, ha='center', fontsize=md, fontweight='normal') 
    x = range(1, len(train_losses)+1)
    axs[0].plot(x, train_losses, label=f'Final train loss: {train_losses[-1]:.4f}')
    axs[0].plot(x, test_losses, label=f'Final test loss: {test_losses[-1]:.4f}')
    axs[0].set_title('Losses', fontsize=md)
    axs[0].set_xlabel('Epoch', fontsize=md)
    axs[0].set_ylabel('Loss', fontsize=md)
    axs[0].legend(fontsize=sm)
    axs[0].tick_params(axis='both', labelsize=sm)
    # Optionally use a logarithmic y-scale
    #axs[0].set_yscale('log')
    axs[0].grid(True, which="both", linestyle='--', linewidth=0.5)
    axs[1].plot(x, train_accuracy, label=f'Final train accuracy: {train_accuracy[-1]:.4f}%')
    axs[1].plot(x, test_accuracy, label=f'Final test accuracy: {test_accuracy[-1]:.4f}%')
    axs[1].set_title('Accuracy', fontsize=md)
    axs[1].set_xlabel('Epoch', fontsize=md)
    axs[1].set_ylabel('Accuracy (%)', fontsize=sm)
    axs[1].legend(fontsize=sm)
    axs[1].tick_params(axis='both', labelsize=sm)
    axs[1].grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(f"{title}.png",dpi=300) 
    plt.show()