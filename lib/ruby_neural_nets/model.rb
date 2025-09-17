require 'ruby_neural_nets/optimizers/constant'

module RubyNeuralNets

  class Model

    # Define a model for processing images and outputting a given number of classes
    #
    # Parameters::
    # * *rows* (Integer): Number of rows per image
    # * *cols* (Integer): Number of columns per image
    # * *channels* (Integer): Number of channels per image
    # * *nbr_classes* (Integer): Number of classes to identify
    # * *optimizer* (Optimizer): The optimizer to be used [default: Optimizer::Constant.new(learning_rate: 0.001)]
    def initialize(rows, cols, channels, nbr_classes, optimizer: Optimizer::Constant.new(learning_rate: 0.001))
      @optimizer = optimizer
    end

    # Perform the forward propagation given an input layer
    #
    # Parameters::
    # * *x* (Numo::DFloat): The input layer
    # Result::
    # * Numo::DFloat: The corresponding output layer
    def forward_propagate(x)
      raise 'Not implemented'
    end

    # Perform the gradient descent, given the predicted output and real one.
    # Prerequisite: forward_propagate must be called prior to this.
    #
    # Parameters::
    # * *da* (Numo::DFloat): The loss derivative from the model predicted output
    # * *a* (Numo::DFloat): The predicted output
    # * *y* (Numo::DFloat): The real output
    def gradient_descent(da, a, y)
      raise 'Not implemented'
    end

    # Adapt some parameters from their derivative and eventual optimization techniques.
    # This method could be called in any layer's backward_propagate method to update trainable parameters.
    #
    # Parameters::
    # * *params* (Numo::DFloat): Parameters to update
    # * *dparams* (Numo::DFloat): Corresponding derivatives of those parameters
    # Result::
    # * Numo::DFloat: New parameters to take into account for next epoch
    def learn(params, dparams)
      @optimizer.learn(params, dparams)
    end
    
  end

end
