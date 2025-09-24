module RubyNeuralNets

  # Store parameters that can be trained
  class Parameter

    # Parameter name
    #   String
    attr_reader :name

    # Optimizer decorations
    #   Hash
    attr_reader :optimizer_parameters

    # Parameter tensor shape
    #   Array
    attr_reader :shape

    # Actual values of this parameter
    #   Numo::DFloat
    attr_reader :values

    # Get the last derivative computed from gradient descent
    #   Numo::DFloat
    attr_reader :dparams
    
    # Gradient check indices, used by the trainer to perform gradient checking
    #   Array
    attr_accessor :gradient_check_indices

    # Constructor
    #
    # Parameters::
    # * *shape* (Array): Shape of this parameter
    # * *initializer* (Class): Initializer class
    # * *optimizer* (Optimizer): Optimizer responsible for learning this parameter
    # * *name* (String): Name that can be used for display or search [default: 'P']
    def initialize(shape, initializer, optimizer, name: 'P')
      @shape = shape
      @values = initializer.new_tensor(shape)
      @optimizer = optimizer
      @optimizer_parameters = {}
      @name = name
      optimizer.init_parameter(self)
    end
      
    # Adapt some parameters from their derivative and eventual optimization techniques.
    # This method could be called in any layer's backward_propagate method to update trainable parameters.
    #
    # Parameters::
    # * *dparams* (Numo::DFloat): Corresponding derivatives of those parameters
    def learn(dparams)
      @dparams = dparams
      @values = @optimizer.learn(self, dparams)
    end
    
  end

end
