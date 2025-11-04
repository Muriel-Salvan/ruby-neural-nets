require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that normalizes images to [0,1] range.
    class ImageMagickNormalize < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        image, y = @dataset[index]
        [image.dispatch(0, 0, image.columns, image.rows, Helpers.image_pixels_map(image), true), y]
      end

    end

  end

end
