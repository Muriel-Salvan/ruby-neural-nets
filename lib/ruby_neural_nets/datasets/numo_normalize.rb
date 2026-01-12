require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/sample'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that normalizes Numo DFloat arrays between 0 and 1.
    class NumoNormalize < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Sample: The sample containing input and target data
      def [](index)
        sample = @dataset[index]
        Sample.new(
          -> { sample.input / (2 ** @dataset.image_stats[:depth] - 1) },
          -> { sample.target }
        )
      end

      # Convert an element to an image
      #
      # Parameters::
      # * *element* (Object): The X element returned by [] method
      # Result::
      # * Object: The image representation (delegated to underlying dataset)
      def to_image(element)
        # Scale back by factor before delegating
        @dataset.to_image(element * (2 ** @dataset.image_stats[:depth] - 1))
      end

    end

  end

end
