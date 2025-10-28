require 'rmagick'
require 'ruby_neural_nets/datasets/wrapper'
require 'numo/narray'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies random image transformations for data augmentation.
    # Supports random rotation and resizing.
    class ImageTransform < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be wrapped
      # * *rng* (Random): Random number generator for transformations
      # * *numo_rng* (Numo::Random::Generator): Random number generator for Numo transformations
      # * *rot_angle* (Float): Maximum rotation angle in degrees (rotation will be between -rot_angle and +rot_angle)
      # * *resize* (Array): Array of 2 integers [width, height] for resizing
      # * *noise_intensity* (Float): Intensity of Gaussian noise for transformations
      def initialize(dataset, rng:, numo_rng:, rot_angle:, resize:, noise_intensity:)
        super(dataset)
        @rng = rng
        @numo_rng = numo_rng
        @rot_angle = rot_angle
        @target_width, @target_height = resize
        @noise_intensity = noise_intensity
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

        image = apply_resize(image)
        image = apply_rotate(image)
        image = apply_crop(image)
        image = apply_gaussian_noise(image)

        [image, y]
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

      # Apply rotation transformation
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to rotate
      # Result::
      # * (Magick::Image): Rotated image or original if no rotation needed
      def apply_rotate(image)
        if @rot_angle > 0
          random_angle = @rng.rand(-@rot_angle..@rot_angle)
          image.rotate(random_angle)
        else
          image
        end
      end

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

      # Apply Gaussian noise transformation
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to add noise to
      # Result::
      # * (Magick::Image): Image with Gaussian noise added or original if no noise needed
      def apply_gaussian_noise(image)
        if @noise_intensity > 0
          original_pixels = Numo::DFloat[image.export_pixels]
          image.import_pixels(
            0,
            0,
            image.columns,
            image.rows,
            'RGB',
            (original_pixels + @numo_rng.normal(shape: original_pixels.shape, loc: 0.0, scale: @noise_intensity * 65535)).clip(0, 65535).flatten.to_a
          )
        end
        image
      end

    end

  end

end
