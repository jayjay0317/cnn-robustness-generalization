# CNN Robustness and Generalization under Image Corruption

This project investigates whether standard geometric data augmentation improves the robustness of a convolutional neural network to image corruptions that were not explicitly seen during training.

A baseline CNN and an augmented CNN are trained on CIFAR-10 using the same architecture and training configuration. The augmented model is trained with random cropping and horizontal flipping, while both models are evaluated under identical clean and corrupted test conditions.

The robustness analysis focuses on three corruption types:

- Gaussian noise
- Gaussian blur
- Brightness shifts


## Research Question

Can standard geometric data augmentation improve a CNN's robustness to unseen image corruptions while maintaining clean classification performance?


## Dataset

The experiments use the CIFAR-10 dataset, which contains 60,000 color images across 10 classes.

The original training set was split into:

- 45,000 training images
- 5,000 validation images

The standard 10,000-image test set was used for final clean and corruption evaluation.

Training images were normalized using channel statistics computed from the CIFAR-10 training data:

- Mean: `[0.4914, 0.4822, 0.4465]`
- Standard deviation: `[0.2470, 0.2435, 0.2616]`