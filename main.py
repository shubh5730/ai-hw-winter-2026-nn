"""
Robust Multi-Modal Perception for Autonomous Driving Under Adversarial and Weather Attacks
Course: Advanced AI Topics
Mini Research Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import ViTModel, ViTConfig
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from data_preparation import create_data_loaders, WeatherAugmentation, DrivingDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class VisionEncoder(nn.Module):
    """Vision Transformer encoder for camera images"""
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit.to(device)
    
    def forward(self, images):
        outputs = self.vit(images)
        return outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]

class LiDAREncoder(nn.Module):
    """Simple point cloud encoder for LiDAR data"""
    def __init__(self, input_dim=3, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc = nn.Linear(256, hidden_dim)
    
    def forward(self, point_clouds):
        # point_clouds: [batch_size, num_points, 3]
        x = point_clouds.transpose(1, 2)  # [batch_size, 3, num_points]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = self.fc(x)
        return x.unsqueeze(1)  # [batch_size, 1, hidden_dim]

class MultiModalFusion(nn.Module):
    """Cross-attention fusion for vision and LiDAR features"""
    def __init__(self, vision_dim=768, lidar_dim=256, fusion_dim=512):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.lidar_proj = nn.Linear(lidar_dim, fusion_dim)
        self.cross_attention = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, vision_features, lidar_features):
        # Project to common dimension
        vision_proj = self.vision_proj(vision_features)  # [batch, seq_len, fusion_dim]
        lidar_proj = self.lidar_proj(lidar_features)    # [batch, 1, fusion_dim]
        
        # Cross-attention: vision queries, lidar keys/values
        attended, attention_weights = self.cross_attention(
            vision_proj, lidar_proj, lidar_proj
        )
        
        # Residual connection and normalization
        fused = self.norm(attended + vision_proj)
        return fused, attention_weights

class RobustDrivingModel(nn.Module):
    """Complete multi-modal model for autonomous driving"""
    def __init__(self):
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.lidar_encoder = LiDAREncoder()
        self.fusion = MultiModalFusion()
        self.classifier = nn.Linear(512, 10)  # 10 object classes
    
    def forward(self, camera_images, lidar_data):
        vision_features = self.vision_encoder(camera_images)
        lidar_features = self.lidar_encoder(lidar_data)
        fused_features, attention_weights = self.fusion(vision_features, lidar_features)
        
        # Global average pooling
        pooled = fused_features.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits, attention_weights

class FGSMAttack:
    """Fast Gradient Sign Method adversarial attack"""
    def __init__(self, model, epsilon=0.01):
        self.model = model
        self.epsilon = epsilon
        self.model.eval()
    
    def attack(self, images, labels):
        # Ensure images require gradients
        images = images.clone().detach().requires_grad_(True)
        labels = labels.clone().detach()
        
        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        outputs, _ = self.model(images, torch.zeros(images.size(0), 100, 3).to(device))
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        perturbed = images + self.epsilon * images.grad.sign()
        perturbed = torch.clamp(perturbed, 0, 1)
        
        return perturbed.detach()

class WeatherAugmentation:
    """Simple weather effects simulation"""
    @staticmethod
    def add_rain(image):
        """Add rain effect to image"""
        image_np = image.detach().permute(1, 2, 0).cpu().numpy()
        # Create rain mask
        rain_mask = np.random.random(image_np.shape[:2]) < 0.1
        # Add rain streaks
        for _ in range(50):
            x, y = np.random.randint(0, image_np.shape[1]), np.random.randint(0, image_np.shape[0])
            length = np.random.randint(10, 30)
            for i in range(length):
                if y + i < image_np.shape[0]:
                    image_np[y + i, x] = image_np[y + i, x] * 0.7
        return torch.tensor(image_np).permute(2, 0, 1).to(device)
    
    @staticmethod
    def add_fog(image):
        """Add fog effect to image"""
        image_np = image.detach().permute(1, 2, 0).cpu().numpy()
        fog_layer = np.ones_like(image_np) * 0.3
        fogged = image_np * 0.7 + fog_layer
        return torch.tensor(fogged).permute(2, 0, 1).to(device)

def create_sample_data(batch_size=4):
    """Create sample multi-modal data for testing"""
    # Camera images (random)
    camera_images = torch.rand(batch_size, 3, 224, 224).to(device)
    
    # LiDAR point clouds (random)
    lidar_data = torch.rand(batch_size, 100, 3).to(device)  # 100 points per cloud
    
    # Labels (10 classes)
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    return camera_images, lidar_data, labels

def evaluate_robustness(model, attack, weather_aug):
    """Evaluate model robustness under attacks"""
    print("=== Robustness Evaluation ===")
    
    # Create test data
    images, lidar, labels = create_sample_data(batch_size=8)
    
    # Clean performance
    model.eval()
    with torch.no_grad():
        clean_logits, _ = model(images, lidar)
        clean_acc = (clean_logits.argmax(dim=1) == labels).float().mean().item()
    
    print(f"Clean Accuracy: {clean_acc:.3f}")
    
    # FGSM attack
    model.eval()
    adv_images = attack.attack(images, labels)
    with torch.no_grad():
        adv_logits, _ = model(adv_images, lidar)
        adv_acc = (adv_logits.argmax(dim=1) == labels).float().mean().item()
    
    print(f"FGSM Accuracy: {adv_acc:.3f}")
    
    # Weather augmentation
    weather_images = torch.stack([weather_aug.add_rain(img) for img in images])
    with torch.no_grad():
        weather_logits, _ = model(weather_images, lidar)
        weather_acc = (weather_logits.argmax(dim=1) == labels).float().mean().item()
    
    print(f"Weather Accuracy: {weather_acc:.3f}")
    
    return {
        'clean': clean_acc,
        'fgsm': adv_acc,
        'weather': weather_acc
    }

def visualize_attention(model, image, lidar):
    """Visualize attention weights"""
    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(image.unsqueeze(0), lidar.unsqueeze(0))
    
    # Get attention weights from fusion layer
    attention = attention_weights.squeeze(0).cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title("Input Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(attention, cmap='hot', interpolation='nearest')
    plt.title("Cross-Modal Attention")
    plt.colorbar()
    plt.savefig('attention_visualization.png')
    plt.show()

def main():
    """Main execution function"""
    print("🚗 Robust Multi-Modal Autonomous Driving Perception")
    print("=" * 60)
    
    # Create data loaders
    print("📊 Loading datasets...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=8)
    print(f"✅ Data loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples")
    
    # Initialize model
    model = RobustDrivingModel().to(device)
    print("✅ Model initialized")
    
    # Initialize attack and augmentation
    fgsm_attack = FGSMAttack(model, epsilon=0.03)
    weather_aug = WeatherAugmentation()
    print("✅ Attack and augmentation methods initialized")
    
    # Get a batch of real data
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        images = batch['camera'].to(device)
        lidar = batch['lidar'].to(device)
        labels = batch['label'].to(device)
        
        print(f"✅ Real data batch - Images: {images.shape}, LiDAR: {lidar.shape}, Labels: {labels.shape}")
    
    # Test forward pass with real data
    with torch.no_grad():
        logits, attention = model(images, lidar)
        print(f"✅ Forward pass successful - Output shape: {logits.shape}")
    
    # Evaluate robustness with real data
    results = evaluate_robustness_with_real_data(model, fgsm_attack, weather_aug, test_loader)
    
    # Visualize attention on real sample
    print("\n📊 Generating attention visualization...")
    visualize_attention_on_real_data(model, test_loader)
    
    print("\n🎯 Results Summary:")
    for attack_type, accuracy in results.items():
        print(f"  {attack_type.capitalize():8s}: {accuracy:.3f}")
    
    print("\n✅ Setup complete! Ready for training.")

def evaluate_robustness_with_real_data(model, attack, weather_aug, test_loader):
    """Evaluate model robustness on real test data"""
    print("\n=== Robustness Evaluation on Real Data ===")
    
    model.eval()
    results = {'clean': [], 'fgsm': [], 'weather': []}
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['camera'].to(device)
            lidar = batch['lidar'].to(device)
            labels = batch['label'].to(device)
            
            # Clean performance
            clean_logits, _ = model(images, lidar)
            clean_acc = (clean_logits.argmax(dim=1) == labels).float().mean().item()
            results['clean'].append(clean_acc)
            
            # FGSM attack (on a smaller subset for speed)
            if len(results['fgsm']) < 5:  # Limit for speed
                # Set model to train mode temporarily for gradient computation
                model.train()
                adv_images = attack.attack(images[:4], labels[:4])
                model.eval()
                
                with torch.no_grad():
                    adv_logits, _ = model(adv_images, lidar[:4])
                    adv_acc = (adv_logits.argmax(dim=1) == labels[:4]).float().mean().item()
                    results['fgsm'].append(adv_acc)
            
            # Weather augmentation (on a smaller subset for speed)
            if len(results['weather']) < 5:
                weather_images = torch.stack([weather_aug.add_rain(img) for img in images[:4]])
                weather_logits, _ = model(weather_images, lidar[:4])
                weather_acc = (weather_logits.argmax(dim=1) == labels[:4]).float().mean().item()
                results['weather'].append(weather_acc)
            
            if len(results['clean']) >= 20:  # Limit evaluation for speed
                break
    
    # Calculate averages
    avg_results = {k: np.mean(v) if v else 0.0 for k, v in results.items()}
    
    print(f"Clean Accuracy: {avg_results['clean']:.3f}")
    print(f"FGSM Accuracy: {avg_results['fgsm']:.3f}")
    print(f"Weather Accuracy: {avg_results['weather']:.3f}")
    
    return avg_results

def visualize_attention_on_real_data(model, test_loader):
    """Visualize attention weights on real data"""
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        image = batch['camera'][0].to(device)
        lidar = batch['lidar'][0].to(device)
        label = batch['label'][0].item()
        
        logits, attention_weights = model(image.unsqueeze(0), lidar.unsqueeze(0))
        
        # Get attention weights from fusion layer
        attention = attention_weights.squeeze(0).cpu().numpy()
        
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
        plt.imshow(img_np)
        plt.title(f"Original Image\nLabel: {label}")
        plt.axis('off')
        
        # Attention visualization
        plt.subplot(1, 3, 2)
        plt.imshow(attention, cmap='hot', interpolation='nearest')
        plt.title("Cross-Modal Attention")
        plt.colorbar()
        
        # LiDAR visualization
        plt.subplot(1, 3, 3)
        lidar_np = lidar.cpu().numpy()
        plt.scatter(lidar_np[:, 0], lidar_np[:, 1], c=lidar_np[:, 2], cmap='viridis', s=1)
        plt.title("LiDAR Point Cloud\n(Top View)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.xlim(-20, 20)
        plt.ylim(0, 50)
        
        plt.tight_layout()
        plt.savefig('real_data_attention.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()
