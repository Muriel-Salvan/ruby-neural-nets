require 'ruby_neural_nets/datasets/wrapper'
require 'numo/narray'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies Gaussian noise to images.
    class ImageNoise < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be wrapped
      # * *numo_rng* (Numo::Random::Generator): Random number generator for Numo noise
      # * *noise_intensity* (Float): Intensity of Gaussian noise for transformations
      def initialize(dataset, numo_rng:, noise_intensity:)
        super(dataset)
        @numo_rng = numo_rng
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
        [apply_gaussian_noise(image), y]
      end

      private

      # Apply Gaussian noise transformation
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to add noise to
      # Result::
      # * (Magick::Image): Image with Gaussian noise added or original if no noise needed
      def apply_gaussian_noise(image)
        if @noise_intensity > 0
          original_pixels = Numo::DFloat[image.export_pixels]
          new_image = Magick::Image.new(image.columns, image.rows)
          new_image.import_pixels(
            0,
            0,
            image.columns,
            image.rows,
            'RGB',
            (original_pixels + @numo_rng.normal(shape: original_pixels.shape, loc: 0.0, scale: @noise_intensity * 65535)).clip(0, 65535).flatten.to_a
          )
          new_image
        else
          image
        end
      end

    end

  end

end
