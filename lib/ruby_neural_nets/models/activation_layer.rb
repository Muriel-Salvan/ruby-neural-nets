require 'ruby_neural_nets/models/layer'

module RubyNeuralNets

  module Models

    # Base class for activation layers
    class ActivationLayer < Layer

      # Get the output dimension of this layer
      #
      # Result::
      # * Integer: Output dimension of the layer
      def n_y
        @n_x
      end

    end

  end

end
