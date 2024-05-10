import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix
import seaborn as sns

class NeuralNet(nn.Module):
    def __init__(self, layer_structure):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()

        # Loop through the layer_structure to create each layer
        for i in range(len(layer_structure) - 1):
            self.layers.append(
                nn.Linear(layer_structure[i], layer_structure[i + 1]))

        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply all layers except the last with ReLU activation
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.relu(x)

        # The last layer does not have ReLU if it's an output layer
        x = self.layers[-1](x)
        return x
    
class ConvNet(nn.Module):
    def __init__(self, layers_config):
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
        
    # optimizer = torch.optim.RMSprop(model.parameters())
    for epoch in range(num_epochs):
        batch_losses = []
        batch_correct_predictions = 0
        batch_total_predictions = 0
        for i, (images, labels) in enumerate(train_loader):
            if isCNN:
                images = images.reshape(-1, 1, 28, 28).to(device)
            else:
                images = images.reshape(-1, 28*28).to(device)
            true_labels = labels.to(device)

            # Forward pass and loss calculation
            predicted_labels = model(images)
            loss = loss_function(predicted_labels, true_labels)
            batch_losses.append(loss.item())
            
            # calculate accuracy
            _, prediction_indices = torch.max(predicted_labels, 1)
            _, true_labels_indices = torch.max(true_labels, 1)
            batch_correct_predictions += (prediction_indices ==
                                    true_labels_indices).sum().item()
            batch_total_predictions += true_labels.size(0)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(
                    f'For Epoch [{epoch+1}/{num_epochs}], the Loss is: {loss.item():.4f}')
            
        epoch_test_loss, epoch_test_accuracy = test_model(model, test_loader, device, isCNN, False)
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_accuracy)
        
        if use_scheduler:
            scheduler.step()
        
        # storing epochs' loss and accuracy for plot
        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)
        epoch_accuracy = 100 * (batch_correct_predictions / batch_total_predictions)
        train_accuracies.append(epoch_accuracy)   
         
    return train_losses, train_accuracies, test_losses, test_accuracies    
        
def test_model(model, test_loader, device, isCNN, isBestModel):
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
    
            # max returns (output_value ,index)
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
    matrix = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
    
def show_misclassified_samples(misclassified_samples):
    plt.figure(figsize=(10, 5))
    for i, (image, true_label, predicted_label) in enumerate(misclassified_samples):
        image = image.squeeze()  # Remove channel dimension for plotting
        plt.subplot(2, 5, i + 1)  # Assuming you want to plot 10 samples
        plt.imshow(image.cpu().numpy(), cmap='gray')
        plt.title(f'True: {true_label}, Pred: {predicted_label}')
        plt.axis('off')
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