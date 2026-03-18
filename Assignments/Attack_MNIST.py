import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Data & Configuration

def get_dataloaders(batch_size_train=128, batch_size_test=256):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    
    return train_loader, test_loader


#Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Attack  Logic

def fgsm_perturbation(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    grad = images.grad.data
    perturbed = images + epsilon * grad.sign()
    return torch.clamp(perturbed, 0, 1)

def pgd_perturbation(model, images, labels, epsilon=0.3, alpha=0.01, iters=40):
    perturbed = images.clone().detach()
    for _ in range(iters):
        perturbed.requires_grad = True
        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        grad = perturbed.grad.data
        perturbed = perturbed + alpha * grad.sign()
        
        delta = torch.clamp(perturbed - images, -epsilon, epsilon)
        perturbed = torch.clamp(images + delta, 0, 1).detach()
        
    return perturbed

def mifgsm_perturbation(model, images, labels, epsilon=0.3, alpha=0.01, iters=40, mu=1.0):
    perturbed = images.clone().detach()
    momentum = torch.zeros_like(images).to(images.device)
    
    for _ in range(iters):
        perturbed.requires_grad = True
        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        grad = perturbed.grad.data
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = mu * momentum + grad
        perturbed = perturbed + alpha * momentum.sign()
        
        
        delta = torch.clamp(perturbed - images, -epsilon, epsilon)
        perturbed = torch.clamp(images + delta, 0, 1).detach()
        
    return perturbed


#Traiing 
def train_model(model, train_loader, optimizer, epochs, device):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


#Evaluation
def evaluate_clean(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def evaluate_attack(model, test_loader, attack_func, device, **kwargs):
    model.eval()
    correct = 0
    total = 0
    attacked_success = 0
    originally_correct = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(1)
            mask = clean_preds == labels
            originally_correct += mask.sum().item()

        perturbed_images = attack_func(model, images, labels, **kwargs)

        with torch.no_grad():
            adv_outputs = model(perturbed_images)
            adv_preds = adv_outputs.argmax(1)

        #Calculating Metrics
        correct += (adv_preds == labels).sum().item()
        attacked_success += ((adv_preds != labels) & mask).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    asr = attacked_success / originally_correct if originally_correct > 0 else 0
    return accuracy, asr

#EXecution
if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, test_loader = get_dataloaders()
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("\nTraining Model ")
    train_model(model, train_loader, optimizer, epochs=5, device=device)
    
    # Clean Accuracy
    clean_acc = evaluate_clean(model, test_loader, device)
    print(f"\nClean Test Accuracy: {clean_acc:.2%}")
    
    # Attacks
    print("\nRunning Adversarial Attacks ")
    epsilon_val = 0.3
    
    print("Evaluating FGSM...")
    fgsm_acc, fgsm_asr = evaluate_attack(model, test_loader, fgsm_perturbation, device, epsilon=epsilon_val)
    
    print("Evaluating PGD...")
    pgd_acc, pgd_asr = evaluate_attack(model, test_loader, pgd_perturbation, device, epsilon=epsilon_val)
    
    print("Evaluating MI-FGSM...")
    ifgsm_acc, mifgsm_asr = evaluate_attack(model, test_loader, mifgsm_perturbation, device, epsilon=epsilon_val)
    
    # Results
    print("\n Final Results ")
    print(f"{'Attack Type':<15} | {'Accuracy':<10} | {'ASR':<10}")
    print("-" * 42)
    print(f"{'FGSM':<15} | {fgsm_acc:.2%}     | {fgsm_asr:.2%}")
    print(f"{'PGD':<15} | {pgd_acc:.2%}     | {pgd_asr:.2%}")
    print(f"{'MI-FGSM':<15} | {ifgsm_acc:.2%}     | {mifgsm_asr:.2%}")
