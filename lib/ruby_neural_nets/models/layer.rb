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

      # Adapt some parameters from their derivative and eventual optimization techniques.
      # This method could be called in any layer's backward_propagate method to update trainable parameters.
      #
      # Parameters::
      # * *params* (Numo::DFloat): Parameters to update
      # * *dparams* (Numo::DFloat): Corresponding derivatives of those parameters
      # Result::
      # * Numo::DFloat: New parameters to take into account for next epoch
      def learn(params, dparams)
        @model.learn(params, dparams)
      end
      
    end

  end

end
