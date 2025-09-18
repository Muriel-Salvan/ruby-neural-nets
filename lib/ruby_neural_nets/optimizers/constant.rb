module RubyNeuralNets

  module Optimizers

    class Constant

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
      # * *params* (Numo::DFloat): Parameters to update
      # * *dparams* (Numo::DFloat): Corresponding derivatives of those parameters
      # Result::
      # * Numo::DFloat: New parameters to take into account for next epoch
      def learn(params, dparams)
        new_params = params - @learning_rate * dparams
        puts '[Optimizer/Constant] !!! Learning has invalid values. There is numerical instability. !!!' unless new_params.isfinite.all?
        new_params
      end

    end

  end

end
