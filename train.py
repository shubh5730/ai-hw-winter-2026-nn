"""
Training Pipeline for Robust Multi-Modal Autonomous Driving
Includes training loop, validation, and model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import json
from tqdm import tqdm
import numpy as np

from main import RobustDrivingModel
from data_preparation import create_data_loaders
from attacks import FGSMAttack, PGDAttack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    """Training pipeline for multi-modal model"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        learning_rate=1e-4,
        num_epochs=10,
        save_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # TensorBoard writer
        self.writer = SummaryWriter('runs/robust_driving')
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
        for batch_idx, batch in enumerate(pbar):
            images = batch['camera'].to(device)
            lidar = batch['lidar'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs, _ = self.model(images, lidar)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/Accuracy', 100.*correct/total, step)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')
            for batch in pbar:
                images = batch['camera'].to(device)
                lidar = batch['lidar'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs, _ = self.model(images, lidar)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        
        # Save latest
        path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'✅ Saved best model with validation accuracy: {val_acc:.2f}%')
    
    def train(self):
        """Main training loop"""
        print(f"\n🚀 Starting Training on {device}")
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Train samples: {len(self.train_loader.dataset)}")
        print(f"   Val samples: {len(self.val_loader.dataset)}")
        print("=" * 60)
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(epoch, val_acc, is_best)
        
        print(f"\n🎯 Training Complete!")
        print(f"   Best Validation Accuracy: {self.best_val_acc:.2f}%")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc
        }
        with open('results/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        self.writer.close()
        return history
    
    def test(self):
        """Evaluate on test set"""
        print("\n📊 Testing on Test Set...")
        
        # Load best model
        checkpoint = torch.load(os.path.join(self.save_dir, 'best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                images = batch['camera'].to(device)
                lidar = batch['lidar'].to(device)
                labels = batch['label'].to(device)
                
                outputs, _ = self.model(images, lidar)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        test_acc = 100. * correct / total
        print(f"✅ Test Accuracy: {test_acc:.2f}%")
        
        return test_acc

def main():
    """Main training script"""
    print("🚗 Robust Multi-Modal Autonomous Driving - Training")
    
    
    # Create data loaders
    print(" Loading datasets...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=16,  # Reduced for CPU compatibility
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Create model
    print("Building model...")
    model = RobustDrivingModel()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=1e-4,
        num_epochs=5,
        save_dir='checkpoints'
    )
    
    # Train
    history = trainer.train()
    
    # Test
    test_acc = trainer.test()
    
    # Save final results
    results = {
        'training_history': history,
        'test_accuracy': test_acc,
        'best_val_accuracy': trainer.best_val_acc
    }
    with open('results/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Training pipeline complete!")
    print(f"   Results saved to: results/")
    print(f"   Checkpoints saved to: checkpoints/")

if __name__ == "__main__":
    main()
