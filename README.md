# Fire Detection with ResNet50

A PyTorch-based fire detection system using ResNet50 for binary classification. This repository compares the performance of models trained with and without Unity-generated synthetic fire images.

## Overview

This project implements a deep learning approach to fire detection using a pre-trained ResNet50 model fine-tuned for binary classification (fire/no-fire). The repository includes two comparative approaches:

1. **Standard Training**: Using only real-world fire and no-fire images
2. **Unity-Enhanced Training**: Incorporating Unity-generated synthetic fire images to augment the dataset

## Repository Structure

```
unity_augmentation/
├── sen2fire/                            # Scene-based fire dataset
│   ├── scene1/
│   │   ├── fire/
│   │   └── no_fire/
│   └── scene2/
│       ├── fire/
│       └── no_fire/
├── unity_generated/                     # Unity synthetic fire images
│   ├── area_1_1.png
│   ├── area_1_2.png
│   └── ...
├── .gitattributes                       # Git LFS configuration
├── README.md                            # This file
├── environment.yml                      # Conda environment file (to be added)
├── best_fire_model_no_unity.pth        # Best model without Unity images
├── best_fire_model_with_unity.pth      # Best model with Unity images
├── resnet_with_no_unity.ipynb          # Training notebook without Unity images
└── resnet_with_unity.ipynb             # Training notebook with Unity images
```

## Dataset Structure

The model expects your dataset to be organized as follows:

### Real-world Images (`sen2fire/`)
```
sen2fire/
├── scene1/
│   ├── fire/          # Fire images from scene 1
│   └── no_fire/       # No-fire images from scene 1
├── scene2/
│   ├── fire/          # Fire images from scene 2
│   └── no_fire/       # No-fire images from scene 2
└── ...
```

<sup>†</sup> **Sen2Fire Dataset**: This work uses the *Sen2Fire* dataset available at [https://zenodo.org/records/10881058](https://zenodo.org/records/10881058), originally provided by Rodríguez, Yan, and others. The dataset is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), which permits use, sharing, and adaptation with appropriate credit.

### Unity-Generated Images (`unity_generated/`)
```
unity_generated/
├── fire_001.jpg       # Synthetic fire images
├── fire_002.jpg
└── ...
```

## Environment Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Conda or Miniconda

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd fire-detection
   ```

2. **Create and activate the conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate fire-detection
   ```

3. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Usage

### Option 1: Standard Training (Without Unity Images)

Use the `resnet_with_no_unity.ipynb` notebook for training with only real-world images.

1. **Open the notebook**
   ```bash
   jupyter notebook resnet_with_no_unity.ipynb
   ```

2. **Update dataset paths** in the notebook:
   ```python
   BASE_PATH = "./sen2fire"              # Path to your scene folders
   UNITY_IMAGES_PATH = None              # Disabled for standard training
   ```

3. **Configure training parameters**:
   ```python
   BATCH_SIZE = 32
   NUM_EPOCHS = 50
   max_fire_per_scene = 100      # Limit fire images per scene
   target_fire_ratio = 0.4       # Target 40% fire images
   ```

4. **Run all cells** to train and evaluate the model

### Option 2: Unity-Enhanced Training

Use the `resnet_with_unity.ipynb` notebook for training with Unity-generated synthetic images.

1. **Open the notebook**
   ```bash
   jupyter notebook resnet_with_unity.ipynb
   ```

2. **Update dataset paths** in the notebook:
   ```python
   BASE_PATH = "./sen2fire"
   UNITY_IMAGES_PATH = "./unity_generated"  # Path to Unity images
   ```

3. **Configure Unity integration**:
   ```python
   use_all_unity = True          # Use all Unity images
   target_fire_ratio = 0.5       # Higher ratio with synthetic data
   ```

4. **Run all cells** to train and evaluate the model

## Key Features

### Smart Dataset Sampling
- **Scene-based sampling**: Limits fire images per scene to prevent overfitting
- **Target ratio control**: Maintains desired fire/no-fire ratio
- **Unity integration**: Seamlessly incorporates synthetic fire images

### Advanced Training Features
- **Weighted sampling**: Handles class imbalance automatically
- **Data augmentation**: Random flips, rotations, and color jittering
- **Early stopping**: Prevents overfitting with patience-based stopping
- **Learning rate scheduling**: Reduces learning rate on plateau

### Comprehensive Evaluation
- **Classification reports**: Precision, recall, and F1-scores
- **Confusion matrices**: Visual performance analysis
- **Training curves**: Loss and accuracy plots
- **Model comparison**: Side-by-side performance metrics

## Model Architecture

- **Base Model**: ResNet50 pre-trained on ImageNet
- **Modification**: Final fully connected layer replaced for binary classification
- **Input Size**: 224×224 RGB images
- **Output**: 2 classes (fire/no-fire)

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 32 | Training batch size |
| Learning Rate | 0.0001 | Adam optimizer learning rate |
| Weight Decay | 1e-4 | L2 regularization |
| Early Stopping | 15 epochs | Patience for early stopping |
| LR Scheduler | ReduceLROnPlateau | Learning rate reduction strategy |

## Expected Outputs

After training, you'll get:
- `best_fire_model_no_unity.pth`: Best performing model without Unity images
- `best_fire_model_with_unity.pth`: Best performing model with Unity images
- Training history plots (loss and accuracy curves)
- Classification report with precision, recall, and F1-scores
- Confusion matrix visualization
- Model performance comparison between standard and Unity-enhanced approaches

## Performance Comparison

The notebooks will generate comparative results showing:
- **Accuracy improvements** with synthetic data augmentation
- **Precision/Recall trade-offs** between approaches
- **Training stability** and convergence patterns
- **Generalization performance** on test data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

dastan.nurbekuly@stud.plus.ac.at

---

*For detailed implementation and theoretical background, refer to the notebooks and code comments.*
