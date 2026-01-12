require 'ruby_neural_nets/minibatch'

module RubyNeuralNets

  module Minibatches

    # Minibatch implementation for Numo arrays
    class Numo < Minibatch

      # Get the size of the minibatch
      #
      # Result::
      # * Integer: The minibatch size
      def size
        target.shape[1]
      end

      # Iterate over individual elements of the minibatch
      #
      # Parameters::
      # * *block* (Proc): Block to call for each element
      #   * Parameters::
      #     * *sample* (Sample): The sample being iterated on
      def each_element
        return to_enum(:each_element) unless block_given?
        
        size.times do |i|
          yield Sample.new(
            -> { input[true, i] },
            -> { target[true, i] }
          )
        end
      end

    end

  end

end
