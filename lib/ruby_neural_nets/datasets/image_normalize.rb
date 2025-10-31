require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that normalizes image pixel values to [0, 1]
    class ImageNormalize < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): The dataset to wrap
      # * *minmax_normalize* (Boolean): Whether to apply minmax normalization before dispatching
      def initialize(dataset, minmax_normalize:)
        super(dataset)
        @minmax_normalize = minmax_normalize
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * x: The element X of the dataset
      # * y: The element Y of the dataset
      def [](index)
        image, y = @dataset[index]
        [(@minmax_normalize ? image.normalize_channel(Magick::AllChannels) : image).dispatch(0, 0, image.columns, image.rows, Helpers.image_pixels_map(image), true), y]
      end

    end

  end

end
