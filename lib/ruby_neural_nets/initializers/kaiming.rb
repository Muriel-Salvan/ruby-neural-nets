require 'numo/narray'

module RubyNeuralNets

  module Initializers

    # Initialize tensor with Kaiming normal distribution
    module Kaiming

      # Return a new parameter tensor of a given shape using Kaiming normal initialization
      #
      # Parameters::
      # * *shape* (Array): Tensor's shape
      # Result::
      # * Numo::DFloat: The initialized tensor
      def self.new_tensor(shape)
        # Generate normal distribution with mean 0 and standard deviation for ReLU activation
        Numo::DFloat.new(*shape).rand_norm(0, Math.sqrt(2.0 / shape[1]))
      end

    end

  end

end
