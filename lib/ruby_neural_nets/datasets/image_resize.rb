require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies image resizing.
    class ImageResize < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be wrapped
      # * *resize* (Array): Array of 2 integers [width, height] for resizing
      def initialize(dataset, resize:)
        super(dataset)
        @target_width, @target_height = resize
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
        [apply_resize(image), y]
      end

      # Get some images stats.
      # Those are supposed to be the same for all samples from the dataset and can be used to compute the model's architecture.
      #
      # Result::
      # * Hash: Image stats:
      #   * *rows* (Integer): Number of rows
      #   * *cols* (Integer): Number of columns
      #   * *channels* (Integer): Number of channels
      def image_stats
        {
          rows: @target_height,
          cols: @target_width,
          channels: @dataset.image_stats[:channels]
        }
      end

      private

      # Apply resize transformation if dimensions differ
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to resize
      # Result::
      # * (Magick::Image): Resized image or original if no resize needed
      def apply_resize(image)
        if image.columns != @target_width || image.rows != @target_height
          image.resize(@target_width, @target_height)
        else
          image
        end
      end

    end

  end

end
