require 'numo/narray'
require 'ruby_neural_nets/models/activation_layer'

module RubyNeuralNets

  module Models

    module Layers

      # Dropout layer to prevent overfitting
      class Dropout < ActivationLayer

        # Constructor
        #
        # Parameters::
        # * *rate* (Float): Dropout rate (fraction of units to drop) [default: 0.5]
        def initialize(rate: 0.5)
          @rate = rate
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
          if train
            # Create dropout mask
            mask = (Numo::DFloat.new(input.shape).rand > @rate)
            back_propagation_cache[:mask] = mask
            # Apply dropout and scale
            input * mask / (1 - @rate)
          else
            input
          end
        end

        # Backward propagate an input da (coming from next layers) through this layer.
        #
        # Parameters::
        # * *da* (Numo::DFloat): The input da coming from the next layers
        # Result::
        # * Numo::DFloat: The corresponding layer output da
        def backward_propagate(da)
          da * back_propagation_cache[:mask] / (1 - @rate)
        end

      end

    end

  end

end
