require 'ruby_neural_nets/loss'

module RubyNeuralNets

  module Losses

    # Compute the cross-entropy loss
    class CrossEntropy < Loss

      # Epsilon used to make sure the gradient of matching a and y is never NaN
      Epsilon = 0.0000000001

      # Compute the loss from a predicted output and a real one.
      # No need to divide by numbers of samples, as it will be done by the trainer.
      #
      # Parameters::
      # * *a* (Numo::DFloat): Tensor of predicted output
      # * *y* (Numo::DFloat): Tensor of real expected output
      # Result::
      # * Float: The corresponding loss
      def compute_loss(a, y)
        - (y * Numo::DFloat::Math.log(a)).sum
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
        (a - y) / ((a * (1 - a)) + Epsilon)
      end

    end

  end

end
