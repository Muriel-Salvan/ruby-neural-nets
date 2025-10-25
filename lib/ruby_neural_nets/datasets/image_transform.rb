require 'rmagick'
require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies random image transformations for data augmentation.
    # Currently supports random rotation within a specified angle range.
    class ImageTransform < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be wrapped
      # * *rng* (Random): Random number generator for transformations
      # * *rot_angle* (Float): Maximum rotation angle in degrees (rotation will be between -rot_angle and +rot_angle)
      def initialize(dataset, rng:, rot_angle:)
        super(dataset)
        @rng = rng
        @rot_angle = rot_angle
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

        # Apply random rotation if rot_angle > 0
        if @rot_angle > 0
          # Generate random angle between -rot_angle and +rot_angle
          random_angle = @rng.rand(-@rot_angle..@rot_angle)

          # Apply rotation using ImageMagick
          rotated_image = image.rotate(random_angle)

          # Crop back to original dimensions, centered
          original_width = image.columns
          original_height = image.rows
          transformed_image = rotated_image.crop(
            (rotated_image.columns - original_width) / 2,
            (rotated_image.rows - original_height) / 2,
            original_width,
            original_height
          )
          [transformed_image, y]
        else
          [image, y]
        end
      end

    end

  end

end
