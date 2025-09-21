require 'ruby_neural_nets/helpers'
require 'ruby_neural_nets/optimizer'

module RubyNeuralNets

  module Optimizers

    class Constant < Optimizer

      # Constructor
      #
      # Parameters::
      # * *learning_rate* (Float): Constant learning rate to apply while learning
      def initialize(learning_rate:)
        @learning_rate = learning_rate
      end

      # Adapt some parameters from their derivative and eventual optimization techniques.
      # This method could be called in any layer's backward_propagate method to update trainable parameters.
      #
      # Parameters::
      # * *parameter* (Parameter): Parameters to update
      # * *dparams* (Numo::DFloat): Corresponding derivatives of those parameters
      # Result::
      # * Numo::DFloat: New parameter values to take into account for next epoch
      def learn(parameter, dparams)
        new_params = parameter.values - @learning_rate * dparams
        Helpers.check_instability(new_params)
        new_params
      end

    end

  end

end
