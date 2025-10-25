require 'numo/narray'
require 'ruby_neural_nets/models/activation_layer'

module RubyNeuralNets

  module Models

    module Layers

      # Simple leaky ReLU layer
      class LeakyRelu < ActivationLayer

        # Constructor
        #
        # Parameters::
        # * *negative_coefficient* (Float): The negative coefficient of the leaky ReLu [default: 0.01]
        def initialize(negative_coefficient: 0.01)
          @negative_coefficient = negative_coefficient
        end

        # Forward propagate an input through this layer
        #
        # Parameters::
        # * *input* (Numo::DFloat): The input
        # * *train* (Boolean): Are we in training mode?
        # Result::
        # * Numo::DFloat: The corresponding layer output
        def forward_propagate(input, train)
          back_propagation_cache[:input] = input
          mask = Numo::DFloat.cast(input.gt(0))
          mask * input + (1 - mask) * input * @negative_coefficient
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
          input = back_propagation_cache[:input]
          mask = Numo::DFloat.cast(input.gt(0))
          da * (mask + (1 - mask) * @negative_coefficient)
        end

      end

    end

  end

end
