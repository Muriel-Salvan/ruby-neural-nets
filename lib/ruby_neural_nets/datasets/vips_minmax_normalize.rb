require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/transform_helpers/vips'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies min-max normalization to image pixel values
    class VipsMinmaxNormalize < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        image, y = @dataset[index]
        [TransformHelpers::Vips.minmax_normalize(image), y]
      end

    end

  end

end
