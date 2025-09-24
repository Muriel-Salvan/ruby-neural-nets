require 'numo/narray'
require 'ruby_neural_nets/optimizer'

module RubyNeuralNets

  module Optimizers

    class Adam < Optimizer

      # Constructor
      #
      # Parameters::
      # * *learning_rate* (Float): Constant learning rate to apply while learning
      # * *beta_1* (Float): Momentum weight [default: 0.9]
      # * *beta_2* (Float): RMS prop weight [default: 0.999]
      # * *epsilon* (Float): Stability correction [default: 0.00000001]
      def initialize(learning_rate:, beta_1: 0.9, beta_2: 0.999, epsilon: 0.00000001)
        @learning_rate = learning_rate
        @beta_1 = beta_1
        @beta_2 = beta_2
        @epsilon = epsilon
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
        parameter.optimizer_parameters[:v] = @beta_1 * parameter.optimizer_parameters[:v] + (1 - @beta_1) * dparams
        parameter.optimizer_parameters[:s] = @beta_2 * parameter.optimizer_parameters[:s] + (1 - @beta_2) * dparams ** 2
        parameter.optimizer_parameters[:t] = parameter.optimizer_parameters[:t] + 1
        v_corrected = parameter.optimizer_parameters[:v] / (1 - @beta_1 ** parameter.optimizer_parameters[:t])
        s_corrected = parameter.optimizer_parameters[:s] / (1 - @beta_2 ** parameter.optimizer_parameters[:t])
        learn_from_diff(parameter, @learning_rate * v_corrected / (s_corrected ** 0.5 + @epsilon))
      end

      # Initialize the optimizer's specific parameters of trainable tensors
      #
      # Parameters::
      # * *parameter* (Parameter): The parameter tensoe to initialize
      def init_parameter(parameter)
        parameter.optimizer_parameters.merge!(
          v: Numo::DFloat.zeros(*parameter.shape),
          s: Numo::DFloat.zeros(*parameter.shape),
          t: 0
        )
      end

    end

  end

end
