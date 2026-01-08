require 'numo/narray'
require 'ruby_neural_nets/helpers'
require 'ruby_neural_nets/models/activation_layer'

module RubyNeuralNets

  module Models

    module Layers

      # Simple tanh layer
      class Tanh < ActivationLayer

        # Forward propagate an input through this layer
        #
        # Parameters::
        # * *input* (Numo::DFloat): The input
        # * *train* (Boolean): Are we in training mode?
        # Result::
        # * Numo::DFloat: The corresponding layer output
        def forward_propagate(input, train)
          output = Helpers.tanh(input)
          back_propagation_cache[:output] = output if train
          output
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
          da * (1 - back_propagation_cache[:output] ** 2)
        end

      end

    end

  end

end
