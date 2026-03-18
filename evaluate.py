"""
Comprehensive Evaluation Script
Includes: Robustness Testing, Interpretability Analysis, Result Generation
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from tqdm import tqdm

from main import RobustDrivingModel
from data_preparation import create_data_loaders, WeatherAugmentation
from attacks import FGSMAttack, PGDAttack, CombinedAttack, evaluate_attack_success_rate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RobustnessEvaluator:
    """Comprehensive robustness and interpretability evaluation"""
    
    def __init__(self, model_path='checkpoints/best_model.pth'):
        self.model = RobustDrivingModel().to(device)
        
        # Load trained model
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model from {model_path}")
        else:
            print(f"⚠️ No trained model found, using random weights")
        
        self.model.eval()
        
        # Create attacks
        self.fgsm = FGSMAttack(self.model, epsilon=0.03)
        self.pgd = PGDAttack(self.model, epsilon=0.03, num_steps=10)
        self.weather_aug = WeatherAugmentation()
        self.combined = CombinedAttack(self.model, self.weather_aug)
        
        # Results storage
        self.results = {}
    
    def evaluate_clean_performance(self, test_loader):
        """Evaluate baseline clean performance"""
        print("\n1. Evaluating Clean Performance...")
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Clean Eval"):
                images = batch['camera'].to(device)
                lidar = batch['lidar'].to(device)
                labels = batch['label'].to(device)
                
                outputs, _ = self.model(images, lidar)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100. * correct / total
        print(f"   ✅ Clean Accuracy: {accuracy:.2f}%")
        
        self.results['clean_accuracy'] = accuracy
        return accuracy
    
    def evaluate_adversarial_robustness(self, test_loader, num_batches=10):
        """Evaluate robustness against adversarial attacks"""
        print("\n2. Evaluating Adversarial Robustness...")
        
        fgsm_metrics = {'clean_acc': [], 'adv_acc': [], 'success_rate': []}
        pgd_metrics = {'clean_acc': [], 'adv_acc': [], 'success_rate': []}
        
        for i, batch in enumerate(tqdm(test_loader, desc="Adversarial Eval")):
            if i >= num_batches:
                break
            
            images = batch['camera'].to(device)
            lidar = batch['lidar'].to(device)
            labels = batch['label'].to(device)
            
            # FGSM attack
            adv_images_fgsm = self.fgsm.attack(images, lidar, labels)
            metrics_fgsm = evaluate_attack_success_rate(self.model, images, adv_images_fgsm, lidar, labels)
            fgsm_metrics['clean_acc'].append(metrics_fgsm['clean_accuracy'])
            fgsm_metrics['adv_acc'].append(metrics_fgsm['adversarial_accuracy'])
            fgsm_metrics['success_rate'].append(metrics_fgsm['attack_success_rate'])
            
            # PGD attack
            adv_images_pgd = self.pgd.attack(images, lidar, labels)
            metrics_pgd = evaluate_attack_success_rate(self.model, images, adv_images_pgd, lidar, labels)
            pgd_metrics['clean_acc'].append(metrics_pgd['clean_accuracy'])
            pgd_metrics['adv_acc'].append(metrics_pgd['adversarial_accuracy'])
            pgd_metrics['success_rate'].append(metrics_pgd['attack_success_rate'])
        
        # Average results
        fgsm_avg = {k: np.mean(v) for k, v in fgsm_metrics.items()}
        pgd_avg = {k: np.mean(v) for k, v in pgd_metrics.items()}
        
        print(f"   FGSM Attack:")
        print(f"     Clean Acc: {fgsm_avg['clean_acc']*100:.2f}%")
        print(f"     Adversarial Acc: {fgsm_avg['adv_acc']*100:.2f}%")
        print(f"     Success Rate: {fgsm_avg['success_rate']*100:.2f}%")
        
        print(f"   PGD Attack:")
        print(f"     Clean Acc: {pgd_avg['clean_acc']*100:.2f}%")
        print(f"     Adversarial Acc: {pgd_avg['adv_acc']*100:.2f}%")
        print(f"     Success Rate: {pgd_avg['success_rate']*100:.2f}%")
        
        self.results['fgsm'] = fgsm_avg
        self.results['pgd'] = pgd_avg
        
        return fgsm_avg, pgd_avg
    
    def evaluate_weather_robustness(self, test_loader, num_batches=10):
        """Evaluate robustness under weather conditions"""
        print("\n3. Evaluating Weather Robustness...")
        
        weather_results = {}
        
        for weather_type in ['rain', 'fog', 'snow']:
            correct = 0
            total = 0
            
            for i, batch in enumerate(test_loader):
                if i >= num_batches:
                    break
                
                images = batch['camera'].to(device)
                lidar = batch['lidar'].to(device)
                labels = batch['label'].to(device)
                
                # Apply weather augmentation
                if weather_type == 'rain':
                    weather_images = torch.stack([self.weather_aug.add_rain(img) for img in images])
                elif weather_type == 'fog':
                    weather_images = torch.stack([self.weather_aug.add_fog(img) for img in images])
                else:  # snow
                    weather_images = torch.stack([self.weather_aug.add_snow(img) for img in images])
                
                with torch.no_grad():
                    outputs, _ = self.model(weather_images, lidar)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            accuracy = 100. * correct / total
            weather_results[weather_type] = accuracy
            print(f"   {weather_type.capitalize()}: {accuracy:.2f}%")
        
        self.results['weather'] = weather_results
        return weather_results
    
    def evaluate_combined_attacks(self, test_loader, num_batches=5):
        """Evaluate robustness under combined attacks"""
        print("\n4. Evaluating Combined Attacks (Adversarial + Weather)...")
        
        combined_results = {}
        
        for weather_type in ['rain', 'fog']:
            correct = 0
            total = 0
            
            for i, batch in enumerate(test_loader):
                if i >= num_batches:
                    break
                
                images = batch['camera'].to(device)
                lidar = batch['lidar'].to(device)
                labels = batch['label'].to(device)
                
                # Apply combined attack
                combined_images = self.combined.attack(images, lidar, labels, weather_type)
                
                with torch.no_grad():
                    outputs, _ = self.model(combined_images, lidar)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            accuracy = 100. * correct / total
            combined_results[f'fgsm_{weather_type}'] = accuracy
            print(f"   FGSM + {weather_type.capitalize()}: {accuracy:.2f}%")
        
        self.results['combined'] = combined_results
        return combined_results
    
    def analyze_interpretability(self, test_loader):
        """Analyze model interpretability through attention visualization"""
        print("\n5. Analyzing Interpretability (Attention Mechanisms)...")
        
        # Get sample batch
        batch = next(iter(test_loader))
        images = batch['camera'][:4].to(device)
        lidar = batch['lidar'][:4].to(device)
        labels = batch['label'][:4].to(device)
        
        with torch.no_grad():
            outputs, attention_weights = self.model(images, lidar)
            predictions = outputs.argmax(dim=1)
        
        # Visualize attention for each sample
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        
        for i in range(4):
            # Original image
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Image\nTrue: {labels[i].item()}, Pred: {predictions[i].item()}")
            axes[i, 0].axis('off')
            
            # Attention heatmap
            attention = attention_weights[i].cpu().numpy()
            im = axes[i, 1].imshow(attention, cmap='hot', interpolation='nearest')
            axes[i, 1].set_title("Cross-Modal Attention")
            plt.colorbar(im, ax=axes[i, 1])
            
            # LiDAR visualization
            lidar_np = lidar[i].cpu().numpy()
            axes[i, 2].scatter(lidar_np[:, 0], lidar_np[:, 1], c=lidar_np[:, 2], cmap='viridis', s=1)
            axes[i, 2].set_title("LiDAR Point Cloud")
            axes[i, 2].set_xlabel("X (m)")
            axes[i, 2].set_ylabel("Y (m)")
            axes[i, 2].set_xlim(-20, 20)
            axes[i, 2].set_ylim(0, 50)
        
        plt.tight_layout()
        plt.savefig('results/interpretability_analysis.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved interpretability visualization")
        
        return attention_weights
    
    def generate_robustness_curves(self):
        """Generate robustness comparison curves"""
        print("\n6. Generating Robustness Curves...")
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Attack comparison
        attack_types = ['Clean', 'FGSM', 'PGD']
        accuracies = [
            self.results['clean_accuracy'],
            self.results['fgsm']['adv_acc'] * 100,
            self.results['pgd']['adv_acc'] * 100
        ]
        
        colors = ['green', 'orange', 'red']
        bars1 = ax1.bar(attack_types, accuracies, color=colors, alpha=0.7)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Model Performance Under Adversarial Attacks')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Weather comparison
        weather_types = list(self.results['weather'].keys())
        weather_accs = list(self.results['weather'].values())
        
        bars2 = ax2.bar(['Clean'] + weather_types, 
                       [self.results['clean_accuracy']] + weather_accs,
                       color=['green', 'skyblue', 'lightblue', 'lightsteelblue'],
                       alpha=0.7)
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Model Performance Under Weather Conditions')
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/robustness_curves.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved robustness curves")
    
    def save_results(self):
        """Save all results to JSON"""
        with open('results/evaluation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("\n✅ Results saved to results/evaluation_results.json")
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
       
        print(" COMPREHENSIVE EVALUATION PIPELINE")
        
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Load data
        print("\n Loading test data...")
        _, _, test_loader = create_data_loaders(batch_size=8, num_workers=0)
        
        # Run all evaluations
        self.evaluate_clean_performance(test_loader)
        self.evaluate_adversarial_robustness(test_loader, num_batches=10)
        self.evaluate_weather_robustness(test_loader, num_batches=10)
        self.evaluate_combined_attacks(test_loader, num_batches=5)
        self.analyze_interpretability(test_loader)
        self.generate_robustness_curves()
        self.save_results()
        
        # Print summary
 
        print("📊 EVALUATION SUMMARY")
  
        print(f"Clean Accuracy: {self.results['clean_accuracy']:.2f}%")
        print(f"FGSM Robustness: {self.results['fgsm']['adv_acc']*100:.2f}%")
        print(f"PGD Robustness: {self.results['pgd']['adv_acc']*100:.2f}%")
        print(f"Average Weather Robustness: {np.mean(list(self.results['weather'].values())):.2f}%")
  
        
        return self.results

def main():
    evaluator = RobustnessEvaluator()
    results = evaluator.run_full_evaluation()

if __name__ == "__main__":
    main()
