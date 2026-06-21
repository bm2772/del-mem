import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import structure_classifier_class

#hyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()
num_epochs = 20
batch_size = 32

model = StructureClassifier().to(device)

#returns the average loss over the training dataset for one epoch of training
def train_model(training_dataloader):
    model.train()
    running_loss = 0.0
    for inputs, labels in training_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Training Loss: {running_loss / len(training_dataloader):.4f}')

def validate_model(validation_dataloader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in validation_dataloader:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Validation Loss: {running_loss / len(validation_dataloader):.4f}')
    print(f'Validation Accuracy: {accuracy:.2f}%')

def test_model(test_dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

training_dataloader = DataLoader(StructureDataset(training_data, training_labels), batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(StructureDataset(validation_data, validation_labels), batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(StructureDataset(test_data, test_labels), shuffle=False)
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    train_model(training_dataloader)
    validate_model(validation_dataloader)

#only for visualization purposes, not necessary for training
import matplotlib.pyplot as plt
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
test_model(test_dataloader)

torch.save(model.state_dict(), 'structure_classifier.pth')