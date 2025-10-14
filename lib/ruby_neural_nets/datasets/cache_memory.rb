require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset caching the data of other datasets in memory
    class CacheMemory < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be cached
      def initialize(dataset)
        super
        invalidate
      end

      # Invalidate the cache
      def invalidate
        @cache = {}
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * x: The element X of the dataset
      # * y: The element Y of the dataset
      def [](index)
        @cache[index] = @dataset[index] unless @cache.key?(index)
        @cache[index]
      end

    end

  end

end
