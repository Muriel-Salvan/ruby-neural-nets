require 'numo/narray'

module RubyNeuralNets

  module Initializers

    # Initialize tensor with Xavier Glorot technique on uniform distribution
    module GlorotUniform

      # Return a new parameter tensor of a given shape
      #
      # Parameters::
      # * *shape* (Array): Tensor's shape
      # Result::
      # * Numo::DFloat: The initialized tensor
      def self.new_tensor(shape)
        max_val = Math.sqrt(6.0 / shape.sum)
        Numo::DFloat.new(*shape).rand(-max_val, max_val)
      end

    end

  end

end
