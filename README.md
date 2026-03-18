# Robust Multi-Modal Perception for Autonomous Driving

## Course Topics Integration

This project demonstrates integration of multiple advanced AI topics:

### 🧠 Deep Learning and Neural Nets
- Multi-modal neural architecture design
- Backpropagation through fused sensor inputs
- Loss functions for robustness

### 🎨 Generative AI: Diffusion
- Weather augmentation using diffusion-based techniques
- Realistic weather condition generation

### 🔄 Transformers: Encoders and Decoders
- Vision Transformer (ViT) for image encoding
- Cross-attention mechanisms for multi-modal fusion
- Self-attention within each modality

### 👁️ Applications: Computer Vision
- Object detection and classification
- Multi-modal perception pipeline

### 🔀 Multi-modality
- Camera + LiDAR sensor fusion
- Cross-modal attention mechanisms
- Early/late fusion strategies

### 🛡️ Robust ML and Adversarial Attacks
- FGSM (Fast Gradient Sign Method) attacks
- PGD (Projected Gradient Descent) attacks
- Combined weather + adversarial attacks

### 🚗 Embodied AI: Self-Driving
- End-to-end autonomous driving perception
- Real-world application scenarios

### 🔍 Interpretability and Explainability
- Attention visualization for model decisions
- Cross-modal attention analysis
- Feature importance analysis

### ⚖️ Bias, Fairness, and AI Ethics
- Performance analysis across demographic groups
- Fairness metrics evaluation
- Equitable performance assessment

## Core Research Question

**Can multi-modal transformer-based perception improve robustness and fairness in autonomous driving under diffusion-generated weather and adversarial attacks?**

## Project Structure

```
ai-hw-winter-2026-nn/
├── main.py                 # Main implementation
├── README.md              # This file
├── requirements.txt       # Dependencies
├── data/                  # Dataset directory
├── models/                # Model architectures
├── attacks/               # Attack implementations
├── results/               # Results and visualizations
└── presentation/           # PDF presentation
```

## Installation

### Environment Setup
```bash
# Create virtual environment
py -m venv robust_driving
robust_driving\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers opencv-python numpy pandas matplotlib
```

## Quick Start

```bash
# Run the basic implementation
py main.py
```

## Current Implementation Status

### ✅ Completed
- [x] Multi-modal architecture (Vision + LiDAR)
- [x] Cross-attention fusion mechanism
- [x] FGSM adversarial attack implementation
- [x] Simple weather augmentation (rain, fog)
- [x] Attention visualization
- [x] Basic robustness evaluation

### 🚧 In Progress
- [ ] Real dataset integration (BDD100K, KITTI)
- [ ] Advanced diffusion-based weather generation
- [ ] PGD attack implementation
- [ ] Comprehensive fairness analysis
- [ ] Meta-learning adaptation mechanisms

### 📋 To Do
- [ ] Training on real autonomous driving data
- [ ] Comprehensive evaluation across demographics
- [ ] Advanced interpretability methods
- [ ] Performance optimization
- [ ] PDF presentation creation

## Results Summary

Initial testing with synthetic data shows:
- **Clean Accuracy**: 12.5% (baseline with random weights)
- **FGSM Robustness**: 0% (untrained model vulnerable to attacks)
- **Weather Robustness**: 12.5% (simple augmentation doesn't affect untrained model)

## Next Steps

1. **Dataset Integration**: Download and integrate BDD100K/KITTI datasets
2. **Model Training**: Train the multi-modal model on real data
3. **Advanced Attacks**: Implement PGD and combined attack strategies
4. **Fairness Analysis**: Evaluate performance across demographic groups
5. **Presentation**: Create comprehensive PDF presentation

## Technical Architecture

### Multi-Modal Fusion Pipeline
```
Camera Input (224x224x3) → Vision Transformer → Feature Embeddings (768-dim)
                                                    ↓
LiDAR Input (Nx3) → PointNet Encoder → Feature Embeddings (256-dim)
                                                    ↓
                                            Cross-Attention Fusion
                                                    ↓
                                            Fused Features (512-dim)
                                                    ↓
                                            Classification Head (10 classes)
```

### Attack Pipeline
```
Clean Images → FGSM Attack → Adversarial Images → Model Evaluation
     ↓
Weather Augmentation → Weather-affected Images → Model Evaluation
     ↓
Combined Attacks → Robustness Assessment
```

## Course Learning Objectives Addressed

- ✅ **Understand advanced concepts about AI and ML**: Multi-modal transformers, adversarial robustness
- ✅ **Understand typical applications of advanced techniques**: Autonomous driving, computer vision
- ✅ **Understand how to use advanced algorithms to resolve practical problems**: Robust perception under attacks
- ✅ **Work on mini-research project**: Complete research pipeline from literature to implementation

## References

1. Vaswani et al. "Attention Is All You Need" (Transformer architecture)
2. Dosovitskiy et al. "An Image is Worth 16x16 Words" (Vision Transformers)
3. Goodfellow et al. "Explaining and Harnessing Adversarial Examples" (FGSM)
4. Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD)
5. BDD100K: Diverse Driving Dataset for Autonomous Driving
6. KITTI Vision Benchmark Suite

## Author

Mini Research Project for Advanced AI Topics Course
Winter 2026
