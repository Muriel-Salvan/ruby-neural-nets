require 'numo/narray'

module RubyNeuralNets

  module Initializers

    # Initialize tensor with 0s
    module Zero

      # Return a new parameter tensor of a given shape
      #
      # Parameters::
      # * *shape* (Array): Tensor's shape
      # Result::
      # * Numo::DFloat: The initialized tensor
      def self.new_tensor(shape)
        Numo::DFloat.zeros(*shape)
      end

    end

  end

end
