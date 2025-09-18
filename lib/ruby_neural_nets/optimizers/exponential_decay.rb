require 'ruby_neural_nets/optimizer'

module RubyNeuralNets

  module Optimizers

    class ExponentialDecay < Optimizer

      # Constructor
      #
      # Parameters::
      # * *learning_rate* (Float): Constant learning rate to apply while learning
      # * *decay* (Float): Decay to apply to the learning rate
      def initialize(learning_rate:, decay:)
        @learning_rate = learning_rate
        @decay = decay
      end

      # Adapt some parameters from their derivative and eventual optimization techniques.
      # This method could be called in any layer's backward_propagate method to update trainable parameters.
      #
      # Parameters::
      # * *params* (Numo::DFloat): Parameters to update
      # * *dparams* (Numo::DFloat): Corresponding derivatives of those parameters
      # Result::
      # * Numo::DFloat: New parameters to take into account for next epoch
      def learn(params, dparams)
        learning_rate = @learning_rate * Numo::NMath.exp(-@decay * @idx_epoch).to_f
        puts "[Optimizer/ExponentialDecay] Learning with rate #{learning_rate}"
        new_params = params - dparams * learning_rate
        puts '[Optimizer/ExponentialDecay] !!! Learning has invalid values. There is numerical instability. !!!' unless new_params.isfinite.all?
        new_params
      end

    end

  end

end
