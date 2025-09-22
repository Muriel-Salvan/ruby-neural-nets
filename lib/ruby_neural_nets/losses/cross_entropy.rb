require 'ruby_neural_nets/loss'

module RubyNeuralNets

  module Losses

    # Compute the cross-entropy loss
    class CrossEntropy < Loss

      # Epsilon used to make sure the gradient of matching a and y is never NaN
      Epsilon = 1e-10

      # Compute the loss from a predicted output and a real one.
      #
      # Parameters::
      # * *a* (Numo::DFloat): Tensor of predicted output
      # * *y* (Numo::DFloat): Tensor of real expected output
      # Result::
      # * Numo::DFloat: The corresponding loss
      def compute_loss(a, y)
        - (y * Numo::DFloat::Math.log(a + Epsilon)).sum(axis: 0)
      end

      # Compute the loss gradient from a predicted output and a real one.
      # This will be the first da tensor to be used in the back-propagation.
      #
      # Parameters::
      # * *a* (Numo::DFloat): Tensor of predicted output
      # * *y* (Numo::DFloat): Tensor of real expected output
      # Result::
      # * Numo::DFloat: The corresponding loss gradient
      def compute_loss_gradient(a, y)
        # TODO: Remove old formula if gradient checking is ok
        # (a - y) / ((a * (1 - a)) + Epsilon)
        - y / (a + Epsilon)
      end

    end

  end

end
