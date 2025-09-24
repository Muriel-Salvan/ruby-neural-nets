require 'numo/narray'

module RubyNeuralNets

  module Initializers

    # Initialize tensor with rand numbers between 0 and 1
    module Rand

      # Return a new parameter tensor of a given shape
      #
      # Parameters::
      # * *shape* (Array): Tensor's shape
      # Result::
      # * Numo::DFloat: The initialized tensor
      def self.new_tensor(shape)
        Numo::DFloat.new(*shape).rand
      end

    end

  end

end
