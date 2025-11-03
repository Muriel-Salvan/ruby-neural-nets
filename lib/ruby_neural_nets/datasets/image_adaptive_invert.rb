require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies adaptive image color inversion.
    # Inverts the image's colors if the top left pixel has an intensity in the lower half range.
    class ImageAdaptiveInvert < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        image, y = @dataset[index]
        [apply_adaptive_invert(image), y]
      end

      private

      # Apply adaptive invert transformation to the image
      #
      # Parameters::
      # * *image* (Magick::Image): Input image to potentially invert
      # Result::
      # * Magick::Image: Image with colors inverted if top left pixel intensity is in lower half
      def apply_adaptive_invert(image)
        # Invert if intensity is in lower half range
        if image.pixel_color(0, 0).intensity < 32768
          inverted_image = image.copy
          inverted_image.alpha(Magick::DeactivateAlphaChannel)
          inverted_image.negate
        else
          image
        end
      end

    end

  end

end
