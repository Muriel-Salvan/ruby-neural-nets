require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/transform_helpers/vips'
require 'ruby_neural_nets/sample'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies adaptive image color inversion.
    # Inverts the image's colors if the top left pixel has an intensity in the lower half range.
    class VipsAdaptiveInvert < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Sample: The sample containing input and target data
      def [](index)
        sample = @dataset[index]
        Sample.new(
          -> { TransformHelpers::Vips.adaptive_invert(sample.input) },
          -> { sample.target }
        )
      end

    end

  end

end
