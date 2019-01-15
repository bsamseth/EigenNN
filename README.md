# Neural Network for Quantum Variational Monte Carlo 

This is an implementation of a neural network for use in quantum VMC. The
reason for this, seeing as there are a multitude of NN implementations
available in Python, is mainly to provide efficient first _and_ second order
gradients. Especially the latter is needed to compute the __Laplacian of the network__, a
key component of VMC calculations. This has shown to be a technical
nightmare/extremely inefficient to compute with automatic differentiation tools
in e.g. TensorFlow. 

The structure is modular and allows to add arbitrary layers and arbitrary
activation functions. It'ts built with Eigen in order to be as efficient as possible, 
although not a lot of emphasis has been placed on optimizing.
