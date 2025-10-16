require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset shuffling dataset before each epoch
    class EpochShuffler < Wrapper

      # Prepare the dataset to be served for a given epoch.
      # This is called before starting an epoch.
      # This can be used to generate some data before hand, or shuffle in a particular way.
      def prepare_for_epoch
        # Get the size again as it could have changed if we selected another partition or filtered it
        @indexes = @dataset.size.times.to_a.shuffle
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

    end

  end

end
