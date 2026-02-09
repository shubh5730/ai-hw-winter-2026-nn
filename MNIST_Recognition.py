import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

#Data Prep

# Transformation for Training (with Augmentation)
train_transform = transforms.Compose([
    transforms.RandomRotation(10),        # Rotate +/- 10 degrees
    transforms.RandomAffine(0, translate=(0.1, 0.1)), # Shift up to 10%
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean/std
])

# Transformation for Testing (No Augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                           download=True, transform=train_transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                          download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Models

# A: Shallow MLP 
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# B: Convolutional Neural Network (CNN) 
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 14x14 -> 7x7
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

# C: Vision Transformer (Encoder)
# We split the 28x28 image into 16 patches of 7x7 pixels
class SimpleViT(nn.Module):
    def __init__(self):
        super(SimpleViT, self).__init__()
        
        # Hyperparameters
        self.patch_size = 7
        self.embed_dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.num_patches = (28 // self.patch_size) ** 2 # (4x4 = 16 patches)
        self.input_dim = self.patch_size * self.patch_size # 7*7 = 49 pixels per patch
        
        # 1. Patch Embedding (Linear projection of flattened patches)
        self.patch_embed = nn.Linear(self.input_dim, self.embed_dim)
        
        # 2. Positional Embedding (Learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_dim))
        
        # 3. Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 5. Output Head
        self.mlp_head = nn.Linear(self.embed_dim, 10)

    def forward(self, x):
        # x shape: [Batch, 1, 28, 28]
        b, c, h, w = x.shape
        
        # Unfold to patches: [Batch, 16, 49]
        # (Split image into 7x7 blocks and flatten the blocks)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(b, -1, self.patch_size * self.patch_size)
        
        # Linear Projection
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Embedding
        x += self.pos_embed
        
        # Transformer Pass
        x = self.transformer(x)
        
        # Extract CLS token output for classification
        cls_out = x[:, 0]
        return self.mlp_head(cls_out)


#  Training Engine

def train_and_evaluate(model, name):
    print(f"\n--- Training {name} ---")
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation Loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f"{name} Test Accuracy: {acc:.2f}%")

#Execution

if __name__ == "__main__":
    # 1. Train MLP
    train_and_evaluate(SimpleMLP(), "MLP")
    
    # 2. Train CNN
    train_and_evaluate(SimpleCNN(), "CNN")
    
    # 3. Train Transformer
    train_and_evaluate(SimpleViT(), "Vision Transformer")
