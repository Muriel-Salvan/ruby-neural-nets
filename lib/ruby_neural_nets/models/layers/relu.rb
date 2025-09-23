require 'ruby_neural_nets/models/activation_layer'

module RubyNeuralNets

  module Models

    module Layers

      # Simple ReLU layer
      class Relu < ActivationLayer

        # Forward propagate an input through this layer
        #
        # Parameters::
        # * *input* (Numo::DFloat): The input
        # Result::
        # * Numo::DFloat: The corresponding layer output
        def forward_propagate(input)
          back_propagation_cache[:input] = input
          input.clip(0, nil)
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
          da * (back_propagation_cache[:input] > 0)
        end

      end

    end

  end

end
