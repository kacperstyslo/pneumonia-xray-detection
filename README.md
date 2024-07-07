# pneumonia-xray-detection

#### Model architecture.
Model architecture implements several key concepts in deep learning:

* Feature Extraction: The convolutional and pooling layers automatically learn to extract 
  relevant features from the chest X-ray images. Early layers might detect simple features 
  like edges, while deeper layers combine these to recognize more complex patterns specific 
  to pneumonia.

* Hierarchical Learning: As data flows through the network, each layer learns to recognize 
  increasingly complex and abstract patterns. This hierarchical learning is a key strength 
  of deep learning.

* Non-linearity: The ReLU activation functions introduce non-linearity, allowing the network 
  to learn complex, non-linear relationships in the data.

* Regularization: Dropout and data augmentation help prevent overfitting, ensuring the model 
  generalizes well to new, unseen data.

* Binary Classification: The final sigmoid activation squashes the output to a probability 
  between 0 and 1, suitable for binary classification (normal vs. pneumonia).

During training, the following processes occur:

* Backpropagation: The network makes predictions on the training data. The difference between 
  these predictions and the true labels (the loss) is used to adjust the network's weights 
  through backpropagation.

* Gradient Descent: The Adam optimizer uses gradient descent to iteratively adjust the weights 
  to minimize the loss function.
