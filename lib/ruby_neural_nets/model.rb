require 'numo/narray'
require 'ruby_neural_nets/parameter'

module RubyNeuralNets

  class Model

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
    def initialize(rows, cols, channels, nbr_classes)
      @parameters = []
    end

    # Initialize the back propagation cache.
    # This is called before forward propagating.
    def initialize_back_propagation_cache
      @back_propagation_cache = {}
    end

    # Link the model to an optimizer.
    # This has to be done by the trainer before training.
    #
    # Parameters::
    # * *optimizer* (Optimizer): The optimizer to attach the model to.
    def link_to_optimizer(optimizer)
      @optimizer = optimizer
      @parameters.each { |param| param.link_to_optimizer(@optimizer) }
    end

    # Get parameters from this model
    #
    # Parameters::
    # * *name* (Regexp or String): Regexp matching the parameter names to fetch (or String for exact name match) [default: /.*/]
    # Result::
    # * Array<Parameter>: List of parameters
    def parameters(name: /.*/)
      filter = name.is_a?(Regexp) ? proc { |p| p.name =~ name } : proc { |p| p.name == name }
      @parameters.select { |select_parameter| filter.call(select_parameter) }
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
    # * *name* (String): Name that can be used for display or search [default: 'P']
    # Result::
    # * Parameter: The corresponding parameters tensor
    def register_parameters(shape, initializer, name: 'P')
      param = Parameter.new(shape, initializer, name:)
      @parameters << param
      param
    end

  end

end
