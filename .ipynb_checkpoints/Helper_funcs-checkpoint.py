# ************************************************************
# File:     Helper_funcs.py
# Author:   Devin Schafthuizen
# Date:     Nov 3rd 2024
# Purpose:  Functions for training and measuring the Performance 
#           of convolutional neural networks using PyTorch
# ************************************************************
import torchvision.models as models
import torch, random, time
import torch.nn as nn
import torch.optim as optim
import seaborn as sn
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models.inception import InceptionOutputs

# Global variables 
accuracy = None
labels = None
y_pred = []
y_true = []
transform = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ************************************************************
# Function:  display_model
# Purpose:   Print summary of the model
# ************************************************************
def display_model(model, input_size):
    size = (3, input_size, input_size)
    print(f'Input size = {size}')
    model = model.to(device)
    summary(model, input_size=size, device=str(device))

# ************************************************************
# Function:  load_datasets
# Purpose:   Split training data into training and validation sets return 
#            dataloaders of different sizes (input to CNN's are different)
# ************************************************************
def load_dataset(img_size, augmented):
    global labels
    global transform

    # Convert images to tensors
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    if augmented == True:
        aug_dataset = datasets.ImageFolder(root='dataset/test-aug', transform=transform)
        aug_train_size = int(0.7 * len(aug_dataset))
        aug_val_size = int(0.2 * len(aug_dataset))
        aug_test_size = len(aug_dataset) - aug_train_size - aug_val_size

        train_dataset, val_dataset, test_dataset = random_split(aug_dataset, [aug_train_size, aug_val_size, aug_test_size])
        labels = aug_dataset.classes
        
    elif augmented == False:
        training_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
        test_dataset = datasets.ImageFolder(root='dataset/test', transform=transform)
            
        # Define the training and validation splits (80% train, 20% val)
        train_size = int(0.8 * len(training_dataset))
        val_size = len(training_dataset) - train_size
            
        # Randomly split the dataset
        train_dataset, val_dataset = random_split(training_dataset, [train_size, val_size])
        labels = test_dataset.classes
        
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, val_loader, test_loader

# ************************************************************
# Function:  load_model
# Purpose:   Load a trained model from a filepath 
# ************************************************************
def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))

# ************************************************************
# Function:  replace_classifier
# Purpose:   Freeze all pre-trained layers of the CNN and replace
#            the classification layer to match our dataset
# ************************************************************
def replace_classifier(model, model_name):
    num_classes = len(labels)

    if model_name in ['AlexNet', 'Mobilenet', 'vgg16']:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, out_features=num_classes)
    elif model_name == 'Inception_v3':
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

# ************************************************************
# Function:  train
# Purpose:   Preform both the forward and backward propagation in training
#            Prints results of each epoch and saves the current best model
#            Early stopping will trigger if no improvements are made in 5 epochs
# ************************************************************
def train(model, train_dataset, val_dataset, best_model_path):
    print(f'Using: [{device}] - [{torch.cuda.get_device_name(0)}]')
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training settings
    max_epochs = 30
    early_stopping = 5
    
    model = model.to(device)
    best_val_acc = 0.0
    epochs_without_update = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_dataset:
            # Move images and labels to the appropriate device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
    
            # Forward pass
            outputs = model(images)
            
            # If outputs is an InceptionOutputs, only use the main output
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits 
            else:
                outputs = model(images)

            # Backward pass and optimization
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
        # Calculate average training loss
        avg_train_loss = running_loss / len(train_dataset)
    
        # Validation loop
        model.eval() 
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Disable gradient calculation for validation
        with torch.no_grad():
            for images, labels in val_dataset:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
    
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
    
        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss / len(val_dataset)
        val_accuracy = correct_predictions / total_predictions
            
        # Save the model if the validation accuracy improves
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_without_update = 0
            torch.save(model.state_dict(), best_model_path)
            print(f'Saving the model with validation accuracy: {best_val_acc:.4f}')
        else:
            epochs_without_update = epochs_without_update + 1
            
        print(f'Epoch [{epoch + 1}/{max_epochs}], Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}')

        if epochs_without_update >= early_stopping:
            print('Early stopping condition meet: terminating training')
            return

# ************************************************************
# Function:  evaluate_accuracy
# Purpose:   puts the model in evaluation mode (no weight updates) and 
#            measures the prediction results against the unseen test dataset.
#            Saves the results for creating the confusion matrix
# ************************************************************
def evaluate_accuracy(model, test_dataset):
    model.eval()
    model = model.to(device)

    global accuracy, y_pred, y_true
    correct = 0
    total = 0
    
    # Disable gradient calculations
    with torch.no_grad():
        for images, labels in test_dataset:
            images, labels = images.to(device), labels.to(device)
            
            # Get model predictions
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Update totals
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # map results for the confusion matrix
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Overall test accuracy: {accuracy:.2f}%')

# ************************************************************
# Function:  evaluate_time
# Purpose:   determines the time required for the model to make 
#            a prediction. 
# ************************************************************
def evaluate_time(model):
    model.to(device)
    model.eval()

    root = './dataset/test/testing-'
    image_list = [root+'a/a00025.png',
                  root+'a/a00001.png',
                  root+'b/b00001.png',
                  root+'b/b00002.png',
                  root+'c/c00001.png',
                  root+'c/c00002.png',
                  root+'d/d00001.png',
                  root+'d/d00002.png',
                  root+'e/e00001.png',
                  root+'e/e00002.png',
                  root+'f/f00001.jpg',
                  root+'f/f00002.png']
    pred_times = []

    # Disable gradient calculations
    with torch.no_grad():
        for image in image_list:
    
            # laod a single image 
            rgb_image = Image.open(image).convert('RGB')
            input_image = transform(rgb_image).unsqueeze(0).to(device)
    
            # Measure time for prediction
            start_time = time.time()

            # Get model predictions
            output = model(input_image)
            _, predicted_class = torch.max(output.data, 1)

            # Stop and save prediction times
            end_time = time.time()
            prediction_time = end_time - start_time
            pred_times.append(prediction_time)
            
            predicted_label = labels[predicted_class.item()]
            print(f'Predicted class: {predicted_label}, Actual label: {image.split("/")[-2]}, time: {prediction_time:.4f}') 
    

    average_time = sum(pred_times) / len(pred_times)
    print(f'Average time for prediction: {average_time:.4f} seconds')

# ************************************************************
# Function:  create_confusion_matrix
# Purpose:   With the results from evaluate_accuracy, create a 
#            confusion matrix from the predicate value against the 
#            real value from the dataset.
# ************************************************************
def create_confusion_matrix(model, test_dataset):
    global accuracy, y_pred, y_true
    if (accuracy == None):
        print('Run evaluate_accuracy before this function')
        return
    
    # initialize confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix by row (i.e., by the number of samples in each class)
    cf_matrix_normalized = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cf_matrix_normalized, index = [i for i in labels], columns = [i for i in labels])
    
    sn.heatmap(df_cm, annot=True, fmt='.0%', cmap='Blues', cbar=False)
    
    # Add labels for the axes
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('Actual Class', fontsize=12)
    
    # Add title below the main graph
    plt.figtext(0.5, 0.02, 'Accuracy = ' + str(round(accuracy, 2)) + '%', ha='center', fontsize=14)
    
    # Move the X-axis labels to the top
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.xticks(rotation=45)
    
    plt.show()