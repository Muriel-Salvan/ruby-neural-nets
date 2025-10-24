require 'numo/narray'
require 'ruby_neural_nets/optimizer'

module RubyNeuralNets

  module Optimizers

    class ExponentialDecay < Optimizer

      # Constructor
      #
      # Parameters::
      # * *learning_rate* (Float): Constant learning rate to apply while learning
      # * *decay* (Float): Decay to apply to the learning rate
      # * *weight_decay* (Float): Weight decay (L2 regularization) coefficient
      def initialize(learning_rate:, decay:, weight_decay:)
        super(weight_decay:)
        @learning_rate = learning_rate
        @decay = decay
        log "learning_rate: #{@learning_rate}, decay: #{@decay}, weight_decay: #{@weight_decay}"
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
        learn_from_diff(parameter, dparams * @learning_rate * Numo::NMath.exp(-@decay * @idx_epoch).to_f)
      end

    end

  end

end
