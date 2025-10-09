require 'numo/narray'
require 'ruby_neural_nets/loss'

module RubyNeuralNets

  module Losses

    # Compute the cross-entropy loss for Torch.rb
    class CrossEntropyTorch < Loss

      # Constructor
      def initialize
        super
        @criterion = ::Torch::NN::CrossEntropyLoss.new
      end

      # Compute the loss from a predicted output and a real one.
      #
      # Parameters::
      # * *a* (Object): Tensor of predicted output
      # * *y* (Object): Tensor of real expected output
      # Result::
      # * Object: The corresponding loss
      def compute_loss(a, y)
        @criterion.call(a, y)
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
        # Nothing to do here: PyTorch will handle it.
      end

    end

  end

end
