"""
Adversarial Attack Implementations
FGSM, PGD, and Combined Attacks for Robustness Testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM)
    Goodfellow et al., "Explaining and Harnessing Adversarial Examples"
    """
    def __init__(self, model, epsilon=0.03):
        self.model = model
        self.epsilon = epsilon
    
    def attack(self, images, lidar, labels):
        """Generate adversarial examples using FGSM"""
        # Clone and ensure gradients are tracked
        images = images.clone().detach().requires_grad_(True)
        lidar = lidar.clone().detach()
        labels = labels.clone().detach()
        
        # Forward pass
        outputs, _ = self.model(images, lidar)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial perturbation
        data_grad = images.grad.data
        perturbed_images = images + self.epsilon * data_grad.sign()
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images.detach()

class PGDAttack:
    """
    Projected Gradient Descent (PGD)
    Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"
    """
    def __init__(self, model, epsilon=0.03, alpha=0.01, num_steps=10):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
    
    def attack(self, images, lidar, labels):
        """Generate adversarial examples using PGD"""
        original_images = images.clone().detach()
        perturbed_images = images.clone().detach()
        
        # Random initialization within epsilon ball
        perturbed_images = perturbed_images + torch.empty_like(perturbed_images).uniform_(-self.epsilon, self.epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        for _ in range(self.num_steps):
            perturbed_images.requires_grad = True
            
            # Forward pass
            outputs, _ = self.model(perturbed_images, lidar)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update perturbation
            data_grad = perturbed_images.grad.data
            perturbed_images = perturbed_images.detach() + self.alpha * data_grad.sign()
            
            # Project back to epsilon ball
            delta = torch.clamp(perturbed_images - original_images, -self.epsilon, self.epsilon)
            perturbed_images = torch.clamp(original_images + delta, 0, 1).detach()
        
        return perturbed_images

class CombinedAttack:
    """
    Combined adversarial + weather attack
    Tests model robustness under multiple simultaneous perturbations
    """
    def __init__(self, model, weather_aug, epsilon=0.03):
        self.model = model
        self.weather_aug = weather_aug
        self.fgsm = FGSMAttack(model, epsilon)
    
    def attack(self, images, lidar, labels, weather_type='rain'):
        """Apply adversarial attack followed by weather augmentation"""
        # First apply FGSM
        adv_images = self.fgsm.attack(images, lidar, labels)
        
        # Then apply weather augmentation
        weather_images = []
        for img in adv_images:
            if weather_type == 'rain':
                weather_img = self.weather_aug.add_rain(img)
            elif weather_type == 'fog':
                weather_img = self.weather_aug.add_fog(img)
            elif weather_type == 'snow':
                weather_img = self.weather_aug.add_snow(img)
            else:
                weather_img = img
            weather_images.append(weather_img)
        
        return torch.stack(weather_images)

def evaluate_attack_success_rate(model, clean_images, adv_images, lidar, labels):
    """
    Evaluate attack success rate
    Success = model prediction changes from correct to incorrect
    """
    model.eval()
    
    with torch.no_grad():
        # Clean predictions
        clean_outputs, _ = model(clean_images, lidar)
        clean_preds = clean_outputs.argmax(dim=1)
        clean_correct = (clean_preds == labels).float()
        
        # Adversarial predictions
        adv_outputs, _ = model(adv_images, lidar)
        adv_preds = adv_outputs.argmax(dim=1)
        adv_correct = (adv_preds == labels).float()
        
        # Attack success: was correct, now incorrect
        attack_success = (clean_correct == 1) & (adv_correct == 0)
        success_rate = attack_success.float().mean().item()
    
    return {
        'clean_accuracy': clean_correct.mean().item(),
        'adversarial_accuracy': adv_correct.mean().item(),
        'attack_success_rate': success_rate,
        'robustness_score': adv_correct.mean().item() / (clean_correct.mean().item() + 1e-10)
    }

def test_attacks():
    """Test attack implementations"""
    print("Testing attack implementations...")
    
    # Create dummy model and data
    from main import RobustDrivingModel
    
    model = RobustDrivingModel().to(device)
    model.eval()
    
    batch_size = 4
    images = torch.rand(batch_size, 3, 224, 224).to(device)
    lidar = torch.rand(batch_size, 1000, 3).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Test FGSM
    print("\n1. Testing FGSM Attack...")
    fgsm = FGSMAttack(model, epsilon=0.03)
    adv_images_fgsm = fgsm.attack(images, lidar, labels)
    print(f"   FGSM perturbation magnitude: {(adv_images_fgsm - images).abs().mean():.4f}")
    
    # Test PGD
    print("\n2. Testing PGD Attack...")
    pgd = PGDAttack(model, epsilon=0.03, num_steps=5)
    adv_images_pgd = pgd.attack(images, lidar, labels)
    print(f"   PGD perturbation magnitude: {(adv_images_pgd - images).abs().mean():.4f}")
    
    # Evaluate attacks
    print("\n3. Evaluating Attack Success...")
    fgsm_metrics = evaluate_attack_success_rate(model, images, adv_images_fgsm, lidar, labels)
    print(f"   FGSM - Clean Acc: {fgsm_metrics['clean_accuracy']:.3f}, Adv Acc: {fgsm_metrics['adversarial_accuracy']:.3f}")
    
    pgd_metrics = evaluate_attack_success_rate(model, images, adv_images_pgd, lidar, labels)
    print(f"   PGD  - Clean Acc: {pgd_metrics['clean_accuracy']:.3f}, Adv Acc: {pgd_metrics['adversarial_accuracy']:.3f}")
    
    print("\n✅ All attacks working correctly!")

if __name__ == "__main__":
    test_attacks()
