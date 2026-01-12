require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/minibatches/numo'

module RubyNeuralNets

  module Datasets

    # Dataset returning data in minibatches.
    # Each element is a Minibatch object containing both X and Y data.
    class Minibatch < Wrapper

      # Get the individual elements dataset
      #   Dataset
      attr_reader :elements_dataset

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset providing elements to be served in minibatches
      # * *max_minibatch_size* (Integer): Max size each minibatch should have [default: 1000]
      def initialize(dataset, max_minibatch_size: 1000)
        super(dataset)
        @max_minibatch_size = max_minibatch_size
        @elements_dataset = dataset
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
      # * Minibatch: The minibatch containing input and target data
      def [](index)
        minibatch = fetch_minibatch(index)
        Minibatches::Numo.new(
          -> { Numo::DFloat.vstack(minibatch.map { |sample| sample.input }).transpose },
          -> { Numo::DFloat[*minibatch.map { |sample| sample.target }].transpose }
        )
      end

      # Return the underlying dataset's label for a given output label of this dataset layer
      #
      # Parameters::
      # * *y* (Object): Label, as returned by the [] method
      # Result::
      # * Object: Corresponding underlying label
      def underlying_label(y)
        @dataset.underlying_label(y.to_a)
      end

      private

      # Fetch the minibatch of (x, y) data points for a given minibatch index
      #
      # Parameters::
      # * *index* (Integer): Minibatch index
      # Result::
      # * Array< Sample >: List of samples for this minibatch
      def fetch_minibatch(index)
        (index * @max_minibatch_size...[(index + 1) * @max_minibatch_size, @dataset.size].min).map { |index| @dataset[index] }
      end

    end

  end

end
