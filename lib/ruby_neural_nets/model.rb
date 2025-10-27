require 'numo/narray'
require 'ruby_neural_nets/parameter'
require 'ruby_neural_nets/logger'

module RubyNeuralNets

  class Model
    include Logger

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
    # * *x* (Object): The input layer
    # * *train* (Boolean): Are we in training mode? [default: false]
    # Result::
    # * Object: The corresponding output layer
    def forward_propagate(x, train: false)
      raise 'Not implemented'
    end

    # Perform the gradient descent, given the predicted output and real one.
    # Prerequisite: forward_propagate must be called prior to this.
    #
    # Parameters::
    # * *da* (Object): The loss derivative from the model predicted output
    # * *a* (Object): The predicted output
    # * *y* (Object): The real output
    # * *loss* (Object): The computed loss
    # * *minibatch_size* (Integer): Minibatch size
    def gradient_descent(da, a, y, loss, minibatch_size)
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
      param = Parameter.new(shape, initializer:, name:)
      @parameters << param
      param
    end

    # Return some model statistics
    #
    # Result::
    # * Hash: Model statistics:
    #   * *parameters* (Hash< String, Hash >): Parameters statistics, per parameter name
    #     * *size* (Integer): Parameter size
    def stats
      {
        parameters: parameters.to_h do |parameter|
          [
            parameter.name,
            {
              size: parameter.size
            }
          ]
        end
      }
    end

  end

end
