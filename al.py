import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim

# Custom dataset class
from sklearn.preprocessing import LabelEncoder

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data[['open', 'high', 'low', 'rate', 'volume', 'volatility', 'sharp', 'signal']]
        self.label_encoder = LabelEncoder()
        self.targets = self.label_encoder.fit_transform(self.data['result'])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        feature = torch.tensor(self.features.iloc[index].values, dtype=torch.float32)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return feature, target


# Create custom dataset
dataset = CustomDataset('merged_4h.csv')

# Create data loader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define model architecture
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 4)
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, targets in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        running_loss += loss.item()
    
    # Print statistics
    accuracy = 100 * correct / total
    print('Epoch: %d | Loss: %.4f | Accuracy: %.2f%%' % (epoch+1, running_loss, accuracy))

# Save trained model
torch.save(model.state_dict(), 'trained_model.pth')
