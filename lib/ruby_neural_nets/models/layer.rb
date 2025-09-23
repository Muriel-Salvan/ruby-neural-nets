module RubyNeuralNets

  module Models

    # Base class for layers
    class Layer

      # Link the layer to a model
      #
      # Parameters::
      # * *model* (Model): Model that is using this layer
      # * *n_x* (Integer): Number of input units
      # * *idx_layer* (Integer): Index of this layer in the model
      def link_to_model(model:, n_x:, idx_layer:)
        @model = model
        @n_x = n_x
        @idx_layer = idx_layer
        initialize_parameters
      end

      # Initialize parameters.
      # This method is optional and is always called once a layer is linked to a model.
      def initialize_parameters
        # By default a layer has no parameters
      end

      # Get the output dimension of this layer
      #
      # Result::
      # * Integer: Output dimension of the layer
      def n_y
        raise 'Not implemented'
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

      # Access the back-propagation cache of this layer
      #
      # Result::
      # * Hash: The layer's back-propagation cache
      def back_propagation_cache
        @model.back_propagation_cache[:layers][@idx_layer]
      end

    end

  end

end
