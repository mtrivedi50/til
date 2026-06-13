# Computer Vision ML

## Summary
Training a CNN model to for action recognition prediction using the UCF101 dataset. The dataset was downloaded from [here](https://www.crcv.ucf.edu/data/UCF101.php).

## Tensorboard
- Gradients:
  - Centered around 0, symmetric
  - Narrowing and stabilizing over time
  - Magnitude between 1e-4 and 1e-2 (neither vanishing nor exploding)
  - Similar magnitude across all layers

- Weights:
  - Gradually spreading out from a narrow initialization
  - Stays centered at zero, symmetric bell shape

- Activations:
  - Pre-ReLU activations are not centered around zero -- TODO: add batch norm

**Note:** only ran for 1 epoch. Training loss = 0.0952, validation loss = 0.0751.

### Scalars

![Scalars](imgs/scalars.png)

### VGG Block 0

![Block 0 Layer 0](imgs/block0.0.png)
![Block 0 Layer 2](imgs/block0.2.png)
![Block 0 Layer 4](imgs/block0.4.png)

### VGG Block 1

![Block 1 Layer 0](imgs/block1.0.png)
![Block 1 Layer 2](imgs/block1.2.png)
![Block 1 Layer 4](imgs/block1.4.png)

### VGG Block 2

![Block 2 Layer 0](imgs/block2.0.png)
![Block 2 Layer 2](imgs/block2.2.png)
![Block 2 Layer 4](imgs/block2.4.png)

### FC Layer 1

![FC Layer 1](imgs/fc1.png)

### FC Layer 2

![FC Layer 2](imgs/fc2.png)

### FC Layer 3

![FC Layer 3](imgs/fc3.png)
