# Fashion-MNIST CNN Classifier

A convolutional neural network (CNN) that classifies grayscale images of clothing items from the Fashion-MNIST dataset into 10 categories.

## Problem Statement

Fashion-MNIST is a drop-in replacement for the classic MNIST digit dataset, but instead of handwritten digits it contains images of clothing (t-shirts, trousers, sneakers, bags, etc). It's a step up in difficulty from MNIST since clothing items have more complex shapes and textures, and several classes (e.g. shirt, T-shirt/top, pullover, coat) look visually similar at low resolution — making it a good next step after digit classification for practicing CNNs.

## Dataset

- **Source:** `tf.keras.datasets.fashion_mnist` (loaded directly via Keras, no manual download needed)
- **Size:** 60,000 training images, 10,000 test images
- **Image format:** 28x28 grayscale
- **Classes (10):** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Model Architecture

A simple CNN with a single convolution block:

```
Conv2D(32 filters, 3x3, ReLU) → MaxPooling2D(2x2) → Flatten → Dense(64, ReLU) → Dense(10, Softmax)
```

- **Conv2D** extracts local features like edges and textures
- **MaxPooling2D** downsamples the feature maps, reducing computation and adding some translation invariance
- **Dense(64)** learns combinations of the extracted features
- **Dense(10, softmax)** outputs a probability distribution over the 10 clothing classes

**Total parameters:** 347,146

## Training

- **Optimizer:** Adam
- **Loss:** Sparse categorical crossentropy
- **Epochs:** 5
- **Validation split:** 10%

## Results

| Metric | Score |
|---|---|
| Training accuracy (final epoch) | 93.2% |
| Validation accuracy (final epoch) | 91.2% |
| **Test accuracy** | **90.8%** |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| T-shirt/top | 0.85 | 0.85 | 0.85 |
| Trouser | 0.99 | 0.98 | 0.98 |
| Pullover | 0.83 | 0.89 | 0.86 |
| Dress | 0.88 | 0.93 | 0.91 |
| Coat | 0.86 | 0.87 | 0.86 |
| Sandal | 0.98 | 0.97 | 0.98 |
| Shirt | 0.81 | 0.68 | 0.74 |
| Sneaker | 0.94 | 0.97 | 0.96 |
| Bag | 0.96 | 0.98 | 0.97 |
| Ankle boot | 0.96 | 0.96 | 0.96 |

### Confusion Matrix Observations

As expected, the model performs very well on visually distinct classes (Trouser, Sandal, Bag, Ankle boot — all 0.96+ F1) but struggles most with **Shirt** (0.74 F1, only 0.68 recall). The confusion matrix shows Shirt is most often confused with T-shirt/top, Pullover, and Coat — these classes look genuinely similar at 28x28 grayscale resolution.

## What I Learned

- How Conv2D and MaxPooling2D layers transform image dimensions as data flows through the network
- The value of a confusion matrix and classification report over a single accuracy number — they reveal *which* classes are hard to distinguish, not just overall performance
- A single conv block is enough to reach ~91% accuracy on Fashion-MNIST, though adding a second conv block would likely help close the gap on the harder classes (Shirt, Pullover, Coat)

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn (confusion matrix, classification report)

## Notes

AI (Claude) was used to help add explanatory comments throughout the notebook and assisted in writing a few lines of code (e.g. the confusion matrix and classification report sections).
