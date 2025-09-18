module RubyNeuralNets

  module Models

    # Base class for layers
    class Layer

      # Constructor
      #
      # Parameters::
      # * *model* (Model): Model that is using this layer
      # * *n_x* (Integer): Number of input units
      def initialize(model:, n_x:)
        @model = model
        @n_x = n_x
        # Setup a cache that can be used by the layer
        @cache = {}
      end

      private

      # Register a new parameters tensor
      #
      # Parameters::
      # * *shape* (Array): Shape of this parameter
      # * *initializer* (Class): Initializer class
      # Result::
      # * Parameter: The corresponding parameters tensor
      def register_parameters(shape, initializer)
        @model.register_parameters(shape, initializer)
      end

    end

  end

end
