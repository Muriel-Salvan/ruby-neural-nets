require 'ruby_neural_nets/minibatch'

module RubyNeuralNets

  module Minibatches

    # Minibatch implementation for Torch tensors
    class Torch < RubyNeuralNets::Minibatch

      # Constructor
      #
      # Parameters::
      # * *x* (::Torch::Tensor): The minibatch input data
      # * *y* (::Torch::Tensor): The minibatch labels
      def initialize(x, y)
        @x = x
        @y = y
      end

      # Get the input data (X) of the minibatch
      #
      # Result::
      # * Object: The minibatch input data
      def x
        @x
      end

      # Get the labels (Y) of the minibatch
      #
      # Result::
      # * Object: The minibatch labels
      def y
        @y
      end

      # Get the size of the minibatch
      #
      # Result::
      # * Integer: The minibatch size
      def size
        @y.shape[0]
      end

      # Iterate over individual elements of the minibatch
      #
      # Parameters::
      # * *block* (Proc): Block to call for each element
      #   * Parameters::
      #     * *element_x* (Object): The X element being iterated on
      #     * *element_y* (Object): The Y element being iterated on
      def each_element(&block)
        return to_enum(:each_element) unless block_given?

        size.times do |i|
          block.call(@x[i], @y[i])
        end
      end

    end

  end

end
