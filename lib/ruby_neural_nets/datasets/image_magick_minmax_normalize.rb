require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/transform_helpers/image_magick'
require 'ruby_neural_nets/sample'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies min-max normalization to image pixel values
    class ImageMagickMinmaxNormalize < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Sample: The sample containing input and target data
      def [](index)
        sample = @dataset[index]
        Sample.new(
          -> { TransformHelpers::ImageMagick.minmax_normalize(sample.input) },
          -> { sample.target }
        )
      end

    end

  end

end
