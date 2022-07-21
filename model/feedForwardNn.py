def sigmoid(z):
    #SIGMOID Compute sigmoid functoon
    #   J = SIGMOID(z) computes the sigmoid of z.
    import numpy as np
    z = np.asarray(z)
    j = 1.0 / (1.0 + np.exp(-z))
    return j


def nnForwardPropergation(nn_params, input_layer_size, hidden_layer_size, output_layer_size, inputs):
    import numpy as np
    #NNCOSTFUNCTION Implements the neural network cost function for a two layer
    #neural network which performs classification
    #   [outputs] = nnForwardPropergation(nn_params, hidden_layer_size, num_labels, ...
    #   inputs) computes the output layer values for the neural network nn_params
    #
    #   inputs should be arranged such that the ith set of inputs is the ith row
    #   outputs is arranged such that the ith row is the ith set of outputs
    #
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    theta1 = np.reshape(nn_params[0 : (hidden_layer_size * (input_layer_size + 1))], 
        (hidden_layer_size, (input_layer_size + 1)) )

    theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)) : ],
        (output_layer_size, hidden_layer_size+1))

    # Setup some useful variables
    inputs = np.atleast_2d(inputs)
    n_inputs = inputs.shape[0]

    # Add column of 1's to X to represent the bias node for each of the n_inputs
    # training examples
    col1s = np.ones((n_inputs,1))
    a1 = np.hstack((col1s, inputs)) 

    #Compute the output from layer 1 and layer 2 where layer 2 is the output
    #layer
    h1 = sigmoid(a1 @ np.transpose(theta1))
    a2 = np.hstack( (col1s, h1) )
    h2 = np.tanh(a2 @ np.transpose(theta2))
    outputs = h2
    return outputs

