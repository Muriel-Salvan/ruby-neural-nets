require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies image grayscale conversion.
    class ImageGrayscale < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * x: The element X of the dataset
      # * y: The element Y of the dataset
      def [](index)
        image, y = @dataset[index]
        [apply_grayscale(image), y]
      end

      # Get some images stats.
      # Those are supposed to be the same for all samples from the dataset and can be used to compute the model's architecture.
      #
      # Result::
      # * Hash: Image stats:
      #   * *rows* (Integer): Number of rows
      #   * *cols* (Integer): Number of columns
      #   * *channels* (Integer): Number of channels (1 for grayscale)
      def image_stats
        @dataset.image_stats.merge(channels: 1)
      end

      private

      # Apply grayscale transformation to the image
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to convert to grayscale
      # Result::
      # * (Magick::Image): Single-channel grayscale image
      def apply_grayscale(image)
        gray_image = image.copy
        gray_image.colorspace = Magick::GRAYColorspace
        gray_image
      end

    end

  end

end
