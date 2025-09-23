require 'ruby_neural_nets/optimizers/constant'
require 'ruby_neural_nets/parameter'

module RubyNeuralNets

  class Model

    # Get all parameters for this model
    #   Array<Parameter>
    attr_reader :parameters

    # Access the back-propagation cache
    #   Hash
    attr_accessor :back_propagation_cache

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
      @parameters = []
      @back_propagation_cache = {}
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

    # Register a new parameters tensor
    #
    # Parameters::
    # * *shape* (Array): Shape of this parameter
    # * *initializer* (Class): Initializer class
    # Result::
    # * Parameter: The corresponding parameters tensor
    def register_parameters(shape, initializer)
      param = Parameter.new(shape, initializer, @optimizer)
      @parameters << param
      param
    end

  end

end
