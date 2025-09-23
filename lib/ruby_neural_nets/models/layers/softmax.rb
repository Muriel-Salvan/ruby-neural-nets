require 'numo/narray'
require 'ruby_neural_nets/helpers'
require 'ruby_neural_nets/models/activation_layer'

module RubyNeuralNets

  module Models

    module Layers

      # Simple softmax layer
      class Softmax < ActivationLayer

        # Forward propagate an input through this layer
        #
        # Parameters::
        # * *input* (Numo::DFloat): The input
        # Result::
        # * Numo::DFloat: The corresponding layer output
        def forward_propagate(input)
          output = Helpers.softmax(input)
          back_propagation_cache[:output] = output
          Helpers.check_instability(output, types: %i[not_finite zero one])
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
          # TODO: Remove old formula if gradient checking is ok
          # da * @cache[:output] * (1 - @cache[:output])
          # Computes dz = J^T * da where J is the softmax Jacobian.
          a = back_propagation_cache[:output]
          sum_term = (a * da).sum(axis: 0)
          a * da - a * sum_term
        end

      end

    end

  end

end
