module RubyNeuralNets

  module Initializers

    # Initialize tensor with Xavier Glorot technique
    module Glorot

      # Return a new parameter tensor of a given shape
      #
      # Parameters::
      # * *shape* (Array): Tensor's shape
      # Result::
      # * Numo::DFloat: The initialized tensor
      def self.new_tensor(shape)
        Numo::DFloat.new(*shape).rand * (2.0 / (shape.sum)) ** 0.5
      end

    end

  end

end
