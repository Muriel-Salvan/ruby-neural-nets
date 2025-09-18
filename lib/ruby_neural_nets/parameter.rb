module RubyNeuralNets

  # Store parameters that can be trained
  class Parameter

    # Optimizer decorations
    #   Hash
    attr_reader :optimizer_parameters

    # Parameter tensor shape
    #   Array
    attr_reader :shape

    # Actual values of this parameter
    #   Numo::DFloat
    attr_reader :values
    
    # Constructor
    #
    # Parameters::
    # * *shape* (Array): Shape of this parameter
    # * *initializer* (Class): Initializer class
    # * *optimizer* (Optimizer): Optimizer responsible for learning this parameter
    def initialize(shape, initializer, optimizer)
      @shape = shape
      @values = initializer.new_tensor(shape)
      @optimizer = optimizer
      @optimizer_parameters = {}
      optimizer.init_parameter(self)
    end
      
    # Adapt some parameters from their derivative and eventual optimization techniques.
    # This method could be called in any layer's backward_propagate method to update trainable parameters.
    #
    # Parameters::
    # * *dparams* (Numo::DFloat): Corresponding derivatives of those parameters
    def learn(dparams)
      @values = @optimizer.learn(self, dparams)
    end
    
  end

end
