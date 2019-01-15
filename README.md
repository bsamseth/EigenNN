# Neural Network for Quantum Variational Monte Carlo 

This is an implementation of a neural network for use in quantum VMC. The
reason for this, seeing as there are a multitude of NN implementations
available in Python, is mainly to provide efficient first _and_ second order
gradients. Especially the latter is needed to compute the __Laplacian of the network__, a
key component of VMC calculations. This has shown to be a technical
nightmare/extremely inefficient to compute with automatic differentiation tools
in e.g. TensorFlow. 

The structure is modular and allows to add arbitrary layers and arbitrary
activation functions. It's built with Eigen in order to be as efficient as possible, 
although not a lot of emphasis has been placed on optimizing.

## Usage Example
The use of this is not seen by it self, but for reference, this is how the module could be used.
```python
import numpy as np
from EigenNN import Dnn
from EigenNN.layer import DenseLayer
from EigenNN.activation import identity, relu, sigmoid

dnn = Dnn()
dnn.add_layer(DenseLayer(2, 50, activation=sigmoid))
dnn.add_layer(DenseLayer(50, 25, activation=relu))
dnn.add_layer(DenseLayer(25, 1))  # Default is identity activation

x = np.random.randn(4, 2)

output = dnn.evaluate(x)  # Shape (4, 1)

# Gradients are summed over the sample rows of x
parameter_gradients = dnn.parameter_gradient(x)  # Shape (1451,)
gradient = dnn.gradient(x)  # Shape (2,)
laplace = dnn.laplace(x)    # Real number
```
