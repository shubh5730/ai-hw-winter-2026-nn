# Robust Multi-Modal Perception for Autonomous Driving

## Core Research Question

**Can multi-modal transformer-based perception improve robustness and fairness in autonomous driving under diffusion-generated weather and adversarial attacks?**

## Project Structure

```
ai-hw-winter-2026-nn/
├── main.py                # Main implementation
├── README.md              # This file
├── requirements.txt       # Dependencies
├── data/                  # Dataset directory
├── models/                # Model architectures
├── attacks/               # Attack implementations
├── results/               # Results and visualizations
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
``

