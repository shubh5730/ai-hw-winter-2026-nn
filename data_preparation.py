"""
Data Preparation Pipeline for Robust Multi-Modal Autonomous Driving
Handles camera images, LiDAR point clouds, and labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DrivingDataset(torch.utils.data.Dataset):
    """
    Multi-modal autonomous driving dataset
    Handles camera images, LiDAR point clouds, and labels
    """
    
    def __init__(
        self, 
        data_dir: str = "data/",
        mode: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        num_points: int = 1000,
        augment: bool = True
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.num_points = num_points
        self.augment = augment
        
        # Define transforms for images
        self.image_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation transforms
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ]) if augment else None
        
        # Create synthetic data if real data doesn't exist
        self.create_synthetic_data()
        
    def create_synthetic_data(self):
        """Create synthetic autonomous driving data for testing"""
        print(f"Creating synthetic {self.mode} data...")
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f"{self.data_dir}/images", exist_ok=True)
        os.makedirs(f"{self.data_dir}/lidar", exist_ok=True)
        os.makedirs(f"{self.data_dir}/labels", exist_ok=True)
        
        # Generate synthetic samples
        num_samples = 1000 if self.mode == "train" else 200
        
        self.samples = []
        for i in range(num_samples):
            # Generate synthetic camera image (road scene)
            image = self.generate_synthetic_image(i)
            image_path = f"{self.data_dir}/images/{self.mode}_{i:04d}.png"
            cv2.imwrite(image_path, image)
            
            # Generate synthetic LiDAR point cloud
            lidar_data = self.generate_synthetic_lidar(i)
            lidar_path = f"{self.data_dir}/lidar/{self.mode}_{i:04d}.npy"
            np.save(lidar_path, lidar_data)
            
            # Generate synthetic label (object class)
            label = random.randint(0, 9)  # 10 classes: 0-9
            label_path = f"{self.data_dir}/labels/{self.mode}_{i:04d}.txt"
            with open(label_path, 'w') as f:
                f.write(str(label))
            
            self.samples.append({
                'image_path': image_path,
                'lidar_path': lidar_path,
                'label': label,
                'weather_condition': random.choice(['clear', 'rain', 'fog', 'snow']),
                'time_of_day': random.choice(['day', 'night', 'dawn', 'dusk'])
            })
        
        print(f"Created {len(self.samples)} synthetic {self.mode} samples")
    
    def generate_synthetic_image(self, idx: int) -> np.ndarray:
        """Generate synthetic road scene image"""
        # Create base road scene
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Sky
        image[:160, :] = [135, 206, 235]  # Sky blue
        
        # Road
        image[160:320, :] = [64, 64, 64]  # Gray road
        
        # Road lines
        for i in range(0, 640, 40):
            cv2.rectangle(image, (i, 238), (i+20, 242), (255, 255, 255), -1)
        
        # Trees/Buildings on sides
        for _ in range(10):
            x = random.randint(0, 640)
            y = random.randint(50, 150)
            cv2.circle(image, (x, y), random.randint(10, 30), (34, 139, 34), -1)
        
        # Add some vehicles
        for _ in range(random.randint(1, 3)):
            x = random.randint(100, 540)
            y = random.randint(200, 280)
            w, h = random.randint(30, 60), random.randint(20, 40)
            cv2.rectangle(image, (x, y), (x+w, y+h), (random.randint(0,255), random.randint(0,255), random.randint(0,255)), -1)
        
        return image
    
    def generate_synthetic_lidar(self, idx: int) -> np.ndarray:
        """Generate synthetic LiDAR point cloud"""
        # Generate points representing road, vehicles, and obstacles
        points = []
        
        # Road surface points
        for _ in range(self.num_points // 2):
            x = random.uniform(-20, 20)  # meters
            y = random.uniform(5, 50)   # meters ahead
            z = random.uniform(-2, 0)   # ground level
            points.append([x, y, z])
        
        # Vehicle/obstacle points
        for _ in range(self.num_points // 4):
            x = random.uniform(-10, 10)
            y = random.uniform(10, 30)
            z = random.uniform(0, 2)
            points.append([x, y, z])
        
        # Background points
        for _ in range(self.num_points // 4):
            x = random.uniform(-50, 50)
            y = random.uniform(5, 100)
            z = random.uniform(-5, 10)
            points.append([x, y, z])
        
        return np.array(points, dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply augmentation if enabled and in training mode
        if self.augment and self.mode == "train" and self.augment_transforms:
            image = self.augment_transforms(image)
        
        # Apply standard transforms
        image = self.image_transforms(image)
        
        # Load LiDAR data
        lidar_data = np.load(sample['lidar_path'])
        
        # Sample fixed number of points
        if len(lidar_data) > self.num_points:
            indices = np.random.choice(len(lidar_data), self.num_points, replace=False)
            lidar_data = lidar_data[indices]
        elif len(lidar_data) < self.num_points:
            # Pad with zeros if not enough points
            padding = np.zeros((self.num_points - len(lidar_data), 3))
            lidar_data = np.vstack([lidar_data, padding])
        
        # Convert to tensor
        lidar_tensor = torch.from_numpy(lidar_data).float()
        
        # Get label
        label = sample['label']
        
        return {
            'camera': image,
            'lidar': lidar_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'weather': sample['weather_condition'],
            'time_of_day': sample['time_of_day'],
            'image_path': sample['image_path']
        }

class WeatherAugmentation:
    """Advanced weather augmentation for robustness testing"""
    
    @staticmethod
    def add_rain(image: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Add rain effect to image"""
        image_np = image.detach().permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # Create rain streaks
        rain_mask = np.zeros_like(image_np)
        num_streaks = int(100 * intensity)
        
        for _ in range(num_streaks):
            x = random.randint(0, image_np.shape[1])
            y = random.randint(0, image_np.shape[0])
            length = random.randint(10, 30)
            thickness = random.randint(1, 2)
            
            for i in range(length):
                if y + i < image_np.shape[0]:
                    rain_mask[y + i, max(0, x-thickness):min(image_np.shape[1], x+thickness)] = 200
        
        # Blend rain with original image
        rainy_image = cv2.addWeighted(image_np, 0.7, rain_mask, 0.3, 0)
        
        # Convert back to tensor
        rainy_image = rainy_image.astype(np.float32) / 255.0
        return torch.from_numpy(rainy_image).permute(2, 0, 1).to(image.device)
    
    @staticmethod
    def add_fog(image: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Add fog effect to image"""
        image_np = image.detach().permute(1, 2, 0).cpu().numpy()
        
        # Create fog layer
        fog_layer = np.ones_like(image_np) * intensity * 0.8
        
        # Apply distance-based fog (more fog at top)
        for i in range(image_np.shape[0]):
            fog_factor = (1 - i / image_np.shape[0]) * intensity
            fog_layer[i, :] *= fog_factor
        
        # Blend fog with original image
        foggy_image = image_np * (1 - intensity * 0.6) + fog_layer
        
        return torch.from_numpy(foggy_image).permute(2, 0, 1).to(image.device)
    
    @staticmethod
    def add_snow(image: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Add snow effect to image"""
        image_np = image.detach().permute(1, 2, 0).cpu().numpy()
        
        # Create snow flakes using numpy (avoid OpenCV issues)
        num_flakes = int(200 * intensity)
        snow_layer = np.copy(image_np)
        
        for _ in range(num_flakes):
            x = random.randint(0, image_np.shape[1] - 1)
            y = random.randint(0, image_np.shape[0] - 1)
            size = random.randint(1, 3)
            
            # Add white snow flakes manually
            for dx in range(-size, size+1):
                for dy in range(-size, size+1):
                    if dx*dx + dy*dy <= size*size:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < image_np.shape[1] and 0 <= ny < image_np.shape[0]:
                            snow_layer[ny, nx] = np.minimum(snow_layer[ny, nx] + 0.5, 1.0)
        
        # Blend snow with original image
        snowy_image = image_np * 0.8 + snow_layer * 0.2
        
        return torch.from_numpy(snowy_image).permute(2, 0, 1).to(image.device)

def create_data_loaders(
    data_dir: str = "data/",
    batch_size: int = 16,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test data loaders"""
    
    # Create datasets
    train_dataset = DrivingDataset(data_dir, mode="train", augment=True)
    val_dataset = DrivingDataset(data_dir, mode="val", augment=False)
    test_dataset = DrivingDataset(data_dir, mode="test", augment=False)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader

def visualize_sample_data(dataset: DrivingDataset, num_samples: int = 4):
    """Visualize sample data from the dataset"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 12))
    
    for i in range(num_samples):
        sample = dataset[i]
        
        # Display camera image
        image = sample['camera'].permute(1, 2, 0).cpu().numpy()
        image = np.clip(image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Camera (Label: {sample['label']}, Weather: {sample['weather']})")
        axes[i, 0].axis('off')
        
        # Display LiDAR point cloud (top view)
        lidar = sample['lidar'].cpu().numpy()
        axes[i, 1].scatter(lidar[:, 0], lidar[:, 1], c=lidar[:, 2], cmap='viridis', s=1)
        axes[i, 1].set_title("LiDAR (Top View)")
        axes[i, 1].set_xlabel("X (m)")
        axes[i, 1].set_ylabel("Y (m)")
        axes[i, 1].set_xlim(-20, 20)
        axes[i, 1].set_ylim(0, 50)
        
        # Display weather-augmented image
        weather_aug = WeatherAugmentation()
        if sample['weather'] == 'rain':
            aug_image = weather_aug.add_rain(sample['camera'])
        elif sample['weather'] == 'fog':
            aug_image = weather_aug.add_fog(sample['camera'])
        elif sample['weather'] == 'snow':
            aug_image = weather_aug.add_snow(sample['camera'])
        else:
            aug_image = sample['camera']
        
        aug_image_np = aug_image.permute(1, 2, 0).cpu().numpy()
        aug_image_np = np.clip(aug_image_np, 0, 1)
        axes[i, 2].imshow(aug_image_np)
        axes[i, 2].set_title(f"Weather Augmented ({sample['weather']})")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Test the data preparation pipeline"""
    print("🚗 Data Preparation Pipeline Test")
    print("=" * 50)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=8)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Test data loading
    print("\nTesting data loading...")
    for batch in train_loader:
        print(f"Camera batch shape: {batch['camera'].shape}")
        print(f"LiDAR batch shape: {batch['lidar'].shape}")
        print(f"Labels shape: {batch['label'].shape}")
        print(f"Weather conditions: {batch['weather']}")
        break
    
    # Visualize sample data
    print("\nGenerating data visualization...")
    visualize_sample_data(train_loader.dataset, num_samples=3)
    
    print("\n✅ Data preparation pipeline working correctly!")

if __name__ == "__main__":
    main()
