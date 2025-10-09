require 'numo/narray'
require 'ruby_neural_nets/parameter'

module RubyNeuralNets

  module Parameters

    # Store parameters used by Torch.rb's models
    class Torch < Parameter

      # Underlying torch parameter
      #   Object
      attr_reader :torch_parameter

      # Constructor
      #
      # Parameters::
      # * *name* (String): Name that can be used for display or search [default: 'P']
      # * *torch_parameter* (Object): The torch parameter
      def initialize(name: 'P', torch_parameter:)
        @torch_parameter = torch_parameter
        super(@torch_parameter.shape, name:)
      end

      # Actual values of this parameter
      #
      # Result::
      # * Numo::DFloat: Parameter values
      def values
        @torch_parameter.numo
      end

    end

  end

end
