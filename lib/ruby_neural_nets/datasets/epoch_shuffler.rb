require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset shuffling dataset before each epoch
    class EpochShuffler < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be shuffled
      # * *rng* (Random): Random number generator for shuffling
      def initialize(dataset, rng: Random.new)
        super(dataset)
        @rng = rng
        randomize_indexes
      end

      # Prepare the dataset to be served for a given epoch.
      # This is called before starting an epoch.
      # This can be used to generate some data before hand, or shuffle in a particular way.
      def prepare_for_epoch
        randomize_indexes
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * x: The element X of the dataset
      # * y: The element Y of the dataset
      def [](index)
        @dataset[@indexes[index]]
      end

      private

      # Randomize indexes
      def randomize_indexes
        # Get the size again as it could have changed if we selected another partition or filtered it
        @indexes = @dataset.size.times.to_a.shuffle(random: @rng)
      end

    end

  end

end
