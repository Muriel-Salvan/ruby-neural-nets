require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies image cropping.
    class ImageCrop < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be wrapped
      # * *crop_size* (Array): Array of 2 integers [width, height] for cropping
      def initialize(dataset, crop_size:)
        super(dataset)
        @target_width, @target_height = crop_size
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
        [apply_crop(image), y]
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

      # Apply crop transformation if image is larger than target
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to crop
      # Result::
      # * (Magick::Image): Cropped image or original if no crop needed
      def apply_crop(image)
        if image.columns > @target_width || image.rows > @target_height
          image.crop((image.columns - @target_width) / 2, (image.rows - @target_height) / 2, @target_width, @target_height)
        else
          image
        end
      end

    end

  end

end
