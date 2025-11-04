require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/transform_helpers/image_magick'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies image grayscale conversion.
    class ImageMagickGrayscale < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        image, y = @dataset[index]
        [TransformHelpers::ImageMagick.grayscale(image), y]
      end

      # Get some images stats.
      # Those are supposed to be the same for all samples from the dataset and can be used to compute the model's architecture.
      #
      # Result::
      # * Hash: Image stats:
      #   * *rows* (Integer or nil): Number of rows if it applies to all images, or nil otherwise
      #   * *cols* (Integer or nil): Number of columns if it applies to all images, or nil otherwise
      #   * *channels* (Integer or nil): Number of channels if it applies to all images, or nil otherwise
      def image_stats
        @dataset.image_stats.merge(channels: 1)
      end

    end

  end

end
