require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/transform_helpers'

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
        [TransformHelpers.adaptive_invert(image), y]
      end

    end

  end

end
