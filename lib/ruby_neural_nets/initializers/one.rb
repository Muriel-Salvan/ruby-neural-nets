module RubyNeuralNets

  module Initializers

    # Initialize tensor with 1s
    module One

      # Return a new parameter tensor of a given shape
      #
      # Parameters::
      # * *shape* (Array): Tensor's shape
      # Result::
      # * Numo::DFloat: The initialized tensor
      def self.new_tensor(shape)
        Numo::DFloat.ones(*shape)
      end

    end

  end

end
