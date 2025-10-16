require 'numo/narray'

module RubyNeuralNets

  module Initializers

    # Initialize tensor with Xavier Glorot technique on normal distribution
    module GlorotNormal

      # Return a new parameter tensor of a given shape
      #
      # Parameters::
      # * *shape* (Array): Tensor's shape
      # Result::
      # * Numo::DFloat: The initialized tensor
      def self.new_tensor(shape)
        Numo::DFloat.new(*shape).rand_norm(0, Math.sqrt(2.0 / shape.sum))
      end

    end

  end

end
