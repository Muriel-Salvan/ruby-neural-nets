require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies image trimming while preserving aspect ratio.
    class ImageTrim < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be wrapped
      def initialize(dataset)
        super(dataset)
        # Store original image stats for aspect ratio calculation
        @original_stats = dataset.image_stats
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        image, y = @dataset[index]
        [apply_trim(image), y]
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
        {
          channels: @dataset.image_stats[:channels]
        }
      end

      private

      # Apply trimming transformation while preserving aspect ratio
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to trim
      # Result::
      # * Magick::Image: Trimmed image with aspect ratio preserved
      def apply_trim(image)
        # Store original aspect ratio
        original_aspect_ratio = image.columns.to_f / image.rows
        # Get the bounding box that we want to trim
        bounding_box = image.bounding_box

        # Calculate new dimensions to restore aspect ratio
        trimmed_width = bounding_box.width
        trimmed_height = bounding_box.height
        # Compute the desired height we want for this trimmed width
        desired_trimmed_height = (trimmed_width.to_f / original_aspect_ratio).round
        if desired_trimmed_height > trimmed_height
          # Add rows
          bounding_box.y -= (desired_trimmed_height - trimmed_height) / 2
          bounding_box.height = desired_trimmed_height
        else
          # Add columns
          desired_trimmed_width = (trimmed_height.to_f * original_aspect_ratio).round
          if desired_trimmed_width > trimmed_width
            bounding_box.x -= (desired_trimmed_width - trimmed_width) / 2
            bounding_box.width = desired_trimmed_width
          end
        end

        if bounding_box.x != 0 || bounding_box.y != 0 || bounding_box.width != image.columns || bounding_box.height != image.rows
          image.crop(bounding_box.x, bounding_box.y, bounding_box.width, bounding_box.height)
        else
          image
        end
      end

    end

  end

end
