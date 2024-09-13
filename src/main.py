import torch
import torch.nn as nn
import torch.optim as optim
from preprocessor import Preprocessor
from data_loader import TamilCharacterDataLoader
from model import CustomCNN

def train_model(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(trainloader.dataset)

def validate_model(model, valloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
    
    val_loss = running_loss / len(valloader.dataset)
    accuracy = correct.double() / len(valloader.dataset)
    return val_loss, accuracy

def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize Preprocessor
    preprocessor = Preprocessor()
    transform = preprocessor.get_transform()

    # Initialize DataLoader
    data_loader = TamilCharacterDataLoader(
        train_dir='src/Dataset/processed/train',
        test_dir='src/Dataset/processed/test',
        transform=transform
    )

    trainloader, valloader, testloader = data_loader.load_data()

    # Get classes for mapping
    classes = data_loader.get_classes(testloader.dataset, 'src/Dataset/processed/TamilChar.csv')
    print("Class Mappings:")
    print(classes)

    # Initialize model, loss function, and optimizer
    model = CustomCNN(num_classes=156).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_model(model, trainloader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_model(model, valloader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Evaluate on test set
    test_loss, test_accuracy = validate_model(model, testloader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'tamil_char_recognition_model.pth')

    # Example prediction display using the classes
    example_inputs, example_labels = next(iter(testloader))
    example_inputs = example_inputs.to(device)
    model.eval()
    with torch.no_grad():
        example_outputs = model(example_inputs)
        _, example_preds = torch.max(example_outputs, 1)
        example_preds = example_preds.cpu().numpy()
        example_labels = example_labels.numpy()
        print("Example Predictions vs Actual Labels:")
        for pred, label in zip(example_preds, example_labels):
            print(f"Predicted: {classes[pred]}, Actual: {classes[label]}")

main()
