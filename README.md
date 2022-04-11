# Implementation-of-two-hidden-layers-neural-network-classifier-from-scratch-in-JAX

To approach the task I first had to read in depth about the JAX library, and deeper understanding led me to XLA (the accelerated linaer algebra)
api and the autograd functionality of JAX. 

- The MNSIT dataset was downlaoded locally andsubswequently loaded onto pytorch dataloaders.
- For the formulation of the multi layer perceptron, a two layer network was chosen which takes in flattened MNIST images (done using np.ravel).In my     -model, the layer has 784 neurons, the first hidden layer has 512 neurons, the second hidden layer has 256 neurons,and finally the last layer performs a softmax on a 10 neuron layer to get to the output
- For the data loading functionality, the custom_collate and custom_transform functions were specified
- Next, the training loop process was formulated. The loss function, accuracy metrics for every epoch and the update functions were explained.

References-
1. https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
2. https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/neural_network_with_tfds_data.ipynb#scrollTo=NwDuFqc9X7ER
3. https://github.com/google/jax/blob/main/examples/mnist_classifier.py
