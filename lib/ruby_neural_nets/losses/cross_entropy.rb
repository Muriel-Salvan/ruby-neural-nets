require 'numo/narray'
require 'ruby_neural_nets/loss'

module RubyNeuralNets

  module Losses

    # Compute the cross-entropy loss
    class CrossEntropy < Loss

      # Weight decay (L2 regularization) coefficient
      attr_reader :weight_decay

      # Constructor
      #
      # Parameters::
      # * *weight_decay* (Float): Weight decay (L2 regularization) coefficient [default: 0.0]
      def initialize(weight_decay: 0.0)
        super()
        @weight_decay = weight_decay
      end

      # Compute the loss from a predicted output and a real one.
      #
      # Parameters::
      # * *a* (Numo::DFloat): Tensor of predicted output
      # * *y* (Numo::DFloat): Tensor of real expected output
      # * *model* (Model): The model
      # Result::
      # * Numo::DFloat: The corresponding loss
      def compute_loss(a, y, model)
        # Add L2 regularization term if weight decay is configured
        - (y * Numo::DFloat::Math.log(a)).sum(axis: 0) +
          0.5 * @weight_decay * model.parameters.sum { |param| (param.values ** 2).sum }
      end

      # Compute the loss gradient from a predicted output and a real one.
      # This will be the first da tensor to be used in the back-propagation.
      #
      # Parameters::
      # * *a* (Numo::DFloat): Tensor of predicted output
      # * *y* (Numo::DFloat): Tensor of real expected output
      # * *model* (Model): The model
      # Result::
      # * Numo::DFloat: The corresponding loss gradient
      def compute_loss_gradient(a, y, model)
        # L2 regularization affects parameters, not the output activation
        # The optimizer handles L2 gradients separately
        - y / a
      end

    end

  end

end
