require 'numo/narray'
require 'ruby_neural_nets/optimizer'

module RubyNeuralNets

  module Optimizers

    # Adam optimizer to be sued with Torch.rb
    class AdamTorch < Optimizer

      # Constructor
      #
      # Parameters::
      # * *learning_rate* (Float): Constant learning rate to apply while learning
      # * *weight_decay* (Float): Weight decay (L2 regularization) coefficient
      # * *beta_1* (Float): Momentum weight [default: 0.9]
      # * *beta_2* (Float): RMS prop weight [default: 0.999]
      # * *epsilon* (Float): Stability correction [default: 0.00000001]
      def initialize(learning_rate:, weight_decay:, beta_1: 0.9, beta_2: 0.999, epsilon: 0.00000001)
        super(weight_decay:)
        @learning_rate = learning_rate
        @beta_1 = beta_1
        @beta_2 = beta_2
        @epsilon = epsilon
        log "learning_rate: #{@learning_rate}, weight_decay: #{@weight_decay}, beta_1: #{@beta_1}, beta_2: #{@beta_2}, epsilon: #{@epsilon}"
      end

      # Teach a given set of parameters
      #
      # Parameters::
      # * *parameters* (Array<Parameter>): Model's parameters that need to be learned
      def teach_parameters(parameters)
        super
        # PyTorch's Adam optimizer handles weight_decay scaling internally
        @torch_optim = ::Torch::Optim::Adam.new(parameters.map(&:torch_parameter), lr: @learning_rate, weight_decay: @weight_decay, betas: [@beta_1, @beta_2], eps: @epsilon)
      end

      # Set the current minibatch being processed
      #
      # Parameters::
      # * *idx_minibatch* (Integer): The minibatch index being processed
      # * *minibatch_size* (Integer): The size of the current minibatch
      def start_minibatch(idx_minibatch, minibatch_size)
        super
        @torch_optim.zero_grad
      end

      # Handle a step after back-propogation
      def step
        @torch_optim.step
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
        # Nothing to do: Torch.rb model won't call it.
      end

      # Initialize the optimizer's specific parameters of trainable tensors
      #
      # Parameters::
      # * *parameter* (Parameter): The parameter tensor to initialize
      def init_parameter(parameter)
        # Nothing to do: Torch.rb model won't call it.
      end

    end

  end

end
