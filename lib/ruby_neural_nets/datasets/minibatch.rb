require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset returning data in minibatches.
    # The y label is an array of the label and the minibatch size.
    class Minibatch < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset providing elements to be served in minibatches
      # * *max_minibatch_size* (Integer): Max size each minibatch should have [default: 1000]
      def initialize(dataset, max_minibatch_size: 1000)
        super(dataset)
        @max_minibatch_size = max_minibatch_size
      end

      # Return the dataset size
      #
      # Result::
      # * Integer: Number of elements in this dataset
      def size
        (@dataset.size / @max_minibatch_size.to_f).ceil
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * x: The element X of the dataset
      # * y: The element Y of the dataset
      def [](index)
        minibatch = fetch_minibatch(index)
        [
          Numo::DFloat[*minibatch.map { |(x, _y)| x }].transpose,
          [
            Numo::DFloat[*minibatch.map { |(_x, y)| y }].transpose,
            minibatch.size
          ]
        ]
      end

      private

      # Fetch the minibatch of (x, y) data points for a given minibatch index
      #
      # Parameters::
      # * *index* (Integer): Minibatch index
      # Result::
      # * Array< [x, y] >: List of x, y data points for this minibatch
      def fetch_minibatch(index)
        (index * @max_minibatch_size...[(index + 1) * @max_minibatch_size, @dataset.size].min).map { |index| @dataset[index] }
      end

    end

  end

end
