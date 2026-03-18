"""
Quick Integration Test
Verifies all components work together before full training
"""

import torch
import sys

def test_imports():
    """Test all imports work"""
    print("Testing imports...")
    try:
        from main import RobustDrivingModel
        from data_preparation import create_data_loaders, WeatherAugmentation
        from attacks import FGSMAttack, PGDAttack
        from tqdm import tqdm
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    print("\nTesting data loading...")
    try:
        from data_preparation import create_data_loaders
        train_loader, val_loader, test_loader = create_data_loaders(batch_size=4, num_workers=0)
        print(f"✅ Data loaders created")
        print(f"   Train: {len(train_loader.dataset)} samples")
        print(f"   Val: {len(val_loader.dataset)} samples")
        print(f"   Test: {len(test_loader.dataset)} samples")
        
        # Test getting a batch
        batch = next(iter(train_loader))
        print(f"✅ Sample batch loaded")
        print(f"   Camera shape: {batch['camera'].shape}")
        print(f"   LiDAR shape: {batch['lidar'].shape}")
        print(f"   Labels shape: {batch['label'].shape}")
        return True, train_loader
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False, None

def test_model():
    """Test model forward pass"""
    print("\nTesting model...")
    try:
        from main import RobustDrivingModel
        model = RobustDrivingModel()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created")
        print(f"   Parameters: {total_params:,}")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        images = torch.rand(2, 3, 224, 224).to(device)
        lidar = torch.rand(2, 1000, 3).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs, attention = model(images, lidar)
        
        print(f"✅ Forward pass successful")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Attention shape: {attention.shape}")
        return True, model
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_attacks(model):
    """Test attack implementations"""
    print("\nTesting attacks...")
    try:
        from attacks import FGSMAttack, PGDAttack
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = torch.rand(2, 3, 224, 224).to(device)
        lidar = torch.rand(2, 1000, 3).to(device)
        labels = torch.randint(0, 10, (2,)).to(device)
        
        # Test FGSM
        fgsm = FGSMAttack(model, epsilon=0.03)
        adv_images_fgsm = fgsm.attack(images, lidar, labels)
        print(f"✅ FGSM attack working")
        print(f"   Perturbation: {(adv_images_fgsm - images).abs().mean():.4f}")
        
        # Test PGD
        pgd = PGDAttack(model, epsilon=0.03, num_steps=3)
        adv_images_pgd = pgd.attack(images, lidar, labels)
        print(f"✅ PGD attack working")
        print(f"   Perturbation: {(adv_images_pgd - images).abs().mean():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Attack test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_weather_augmentation():
    """Test weather augmentation"""
    print("\nTesting weather augmentation...")
    try:
        from data_preparation import WeatherAugmentation
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weather_aug = WeatherAugmentation()
        
        image = torch.rand(3, 224, 224).to(device)
        
        rain_img = weather_aug.add_rain(image)
        fog_img = weather_aug.add_fog(image)
        snow_img = weather_aug.add_snow(image)
        
        print(f"✅ Weather augmentation working")
        print(f"   Rain applied: shape {rain_img.shape}")
        print(f"   Fog applied: shape {fog_img.shape}")
        print(f"   Snow applied: shape {snow_img.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Weather augmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    
    print(" INTEGRATION TEST SUITE")
    
    
    results = []
    
    # Test 1: Imports
    results.append(test_imports())
    if not results[-1]:
        print("\n❌ Critical failure - imports not working")
        return False
    
    # Test 2: Data Loading
    success, train_loader = test_data_loading()
    results.append(success)
    if not success:
        print("\n❌ Critical failure - data loading not working")
        return False
    
    # Test 3: Model
    success, model = test_model()
    results.append(success)
    if not success:
        print("\n❌ Critical failure - model not working")
        return False
    
    # Test 4: Attacks
    results.append(test_attacks(model))
    
    # Test 5: Weather
    results.append(test_weather_augmentation())
    
    # Summary
    
    print(" TEST SUMMARY")
    
    print(f"Imports: {'✅' if results[0] else '❌'}")
    print(f"Data Loading: {'✅' if results[1] else '❌'}")
    print(f"Model: {'✅' if results[2] else '❌'}")
    print(f"Attacks: {'✅' if results[3] else '❌'}")
    print(f"Weather: {'✅' if results[4] else '❌'}")
    
    
    if all(results):
        print("\n✅ ALL TESTS PASSED - Ready for training!")
        return True
    else:
        print("\n⚠️ Some tests failed - review errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
