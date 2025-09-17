require 'numo/narray'
require 'ruby_neural_nets/models/layer'

module RubyNeuralNets

  module Models

    module Layers

      # Simple tanh layer
      class Tanh < Layer

        # Forward propagate an input through this layer
        #
        # Parameters::
        # * *input* (Numo::DFloat): The input
        # Result::
        # * Numo::DFloat: The corresponding layer output
        def forward_propagate(input)
          output = tanh(input)
          @cache[:output] = output
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
          da * (1 - @cache[:output] ** 2)
        end

        private

        # Perform tanh of an array
        #
        # Parameters::
        # * *narray* (Numo::DFloat): The array on which we apply tanh
        # Result::
        # * Numo::DFloat: Resulting tanh
        def tanh(narray)
          exp_array = Numo::DFloat::Math.exp(narray)
          exp_neg_array = Numo::DFloat::Math.exp(-narray)
          (exp_array - exp_neg_array) / (exp_array + exp_neg_array)
        end

      end

    end

  end

end
