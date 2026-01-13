require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that clones each element from the wrapped dataset multiple times.
    # This is useful for data augmentation by duplicating samples.
    class Clone < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be wrapped
      # * *nbr_clones* (Integer): Number of times each element should be cloned
      def initialize(dataset, nbr_clones:)
        super(dataset)
        @nbr_clones = nbr_clones
      end

      # Return the dataset size
      #
      # Result::
      # * Integer: Number of elements in this dataset (original size * nbr_clones)
      def size
        @dataset.size * @nbr_clones
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Sample: The sample containing input and target data
      def [](index)
        @dataset[index / @nbr_clones]
      end

    end

  end

end
