require 'numo/narray'
require 'ruby_neural_nets/loss'

module RubyNeuralNets

  module Losses

    # Compute the cross-entropy loss
    class CrossEntropy < Loss

      # Compute the loss from a predicted output and a real one.
      #
      # Parameters::
      # * *a* (Numo::DFloat): Tensor of predicted output
      # * *y* (Numo::DFloat): Tensor of real expected output
      # * *model* (Model): The model
      # Result::
      # * Numo::DFloat: The corresponding loss
      def compute_loss(a, y, model)
        - (y * Numo::DFloat::Math.log(a)).sum(axis: 0)
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
        - y / a
      end

    end

  end

end
