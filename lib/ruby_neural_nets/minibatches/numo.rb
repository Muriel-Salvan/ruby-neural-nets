require 'ruby_neural_nets/minibatch'

module RubyNeuralNets

  module Minibatches

    # Minibatch implementation for Numo arrays
    class Numo < RubyNeuralNets::Minibatch

      # Constructor
      #
      # Parameters::
      # * *x* (Numo::DFloat): The minibatch input data
      # * *y* (Numo::DFloat): The minibatch labels
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
        @y.shape[1]
      end

      # Iterate over individual elements of the minibatch
      #
      # Parameters::
      # * *block* (Proc): Block to call for each element
      #   * Parameters::
      #     * *x* (Object): The element X being iterated on
      #     * *y* (Object): The element Y being iterated on
      def each_element(&block)
        size.times do |i|
          block.call(@x[true, i], @y[i, true])
        end
      end

    end

  end

end
