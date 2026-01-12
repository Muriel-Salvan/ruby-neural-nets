require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/transform_helpers/image_magick'
require 'ruby_neural_nets/sample'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies image trimming while preserving aspect ratio.
    class ImageMagickTrim < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Sample: The sample containing input and target data
      def [](index)
        sample = @dataset[index]
        Sample.new(
          -> { TransformHelpers::ImageMagick.trim(sample.input) },
          -> { sample.target }
        )
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
        {
          channels: @dataset.image_stats[:channels],
          depth: @dataset.image_stats[:depth]
        }
      end

    end

  end

end
