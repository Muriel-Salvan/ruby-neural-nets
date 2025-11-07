require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that normalizes Numo DFloat arrays by dividing by a factor.
    class NumoNormalize < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be wrapped
      # * *factor* (Float): Normalization factor to divide by
      def initialize(dataset, factor:)
        super(dataset)
        @factor = factor
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        x, y = @dataset[index]
        # Normalize by dividing by factor
        [x / @factor, y]
      end

      # Convert an element to an image
      #
      # Parameters::
      # * *element* (Object): The X element returned by [] method
      # Result::
      # * Object: The image representation (delegated to underlying dataset)
      def to_image(element)
        # Scale back by factor before delegating
        @dataset.to_image(element * @factor)
      end

    end

  end

end
