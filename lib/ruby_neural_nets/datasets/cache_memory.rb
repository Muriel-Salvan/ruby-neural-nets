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
        @cache_image_stats = nil
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        @cache[index] = @dataset[index] unless @cache.key?(index)
        @cache[index]
      end

      # Get some images stats.
      # Those are supposed to be the same for all samples from the dataset and can be used to compute the model's architecture.
      #
      # Result::
      # * Hash: Image stats:
      #   * *rows* (Integer or nil): Number of rows if it applies to all images, or nil otherwise
      #   * *cols* (Integer or nil): Number of columns if it applies to all images, or nil otherwise
      #   * *channels* (Integer or nil): Number of channels if it applies to all images, or nil otherwise
      #   * *depth* (Integer or nil): Depth (number of bits) used to encode pixel channel's values if it applies to all images, or nil otherwise
      def image_stats
        @cache_image_stats = @dataset.image_stats if @cache_image_stats.nil?
        @cache_image_stats
      end

    end

  end

end
