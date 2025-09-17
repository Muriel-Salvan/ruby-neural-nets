require 'numo/narray'
require 'ruby_neural_nets/models/layer'

module RubyNeuralNets

  module Models

    module Layers

      # Dense layer outputing a given number of units
      class Dense < Layer

        # Constructor
        #
        # Parameters::
        # * *model* (Model): Model that is using this layer
        # * *n_x* (Integer): Number of input units
        # * *nbr_units* (Integer): The number of units for this dense layer
        def initialize(model:, n_x:, nbr_units:)
          super(model:, n_x:)
          @nbr_units = nbr_units
          # Layer weights [nbr_units, n_x]
          @w = Numo::DFloat.new(nbr_units, n_x).rand
          # Layer bias [nbr_units, 1]
          @b = Numo::DFloat.zeros(nbr_units, 1)
        end

        # Forward propagate an input through this layer
        #
        # Parameters::
        # * *input* (Numo::DFloat): The input
        # Result::
        # * Numo::DFloat: The corresponding layer output
        def forward_propagate(input)
          @cache[:input] = input
          @w.dot(input) + @b
        end

        # Backward propagate an input da (coming from next layers) through this layer.
        # Some examples of the purpose of this function:
        # * da * g'(z) in the case of an activation layer using a function g.
        # * W.transpose.dot(da) in the case of a dense layer.
        # Learning and optiomization can be called on eventual parameters by calling w = learn(w, dw).
        #
        # Parameters::
        # * *da* (Numo::DFloat): The input da coming from the next layers
        # Result::
        # * Numo::DFloat: The corresponding layer output da
        def backward_propagate(da)
          m = @cache[:input].shape[1]
          @w = learn(@w, da.dot(@cache[:input].transpose) / m)
          @b = learn(@b, da.sum(axis: 1, keepdims: true) / m)
          @w.transpose.dot(da)
        end

      end

    end

  end

end
