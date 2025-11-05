require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that removes the alpha channel from Vips images.
    # If the image has 4 bands (RGBA), it removes the alpha channel to get 3 bands (RGB).
    # If the image has fewer bands, it leaves the image unchanged.
    class VipsRemoveAlpha < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        image, y = @dataset[index]
        [image.bands == 4 ? image.extract_band(0, n: 3) : image, y]
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
        @dataset.image_stats.merge(channels: 3)
      end

    end

  end

end
