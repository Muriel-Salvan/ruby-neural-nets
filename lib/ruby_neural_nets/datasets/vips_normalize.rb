require 'ruby_neural_nets/datasets/wrapper'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that normalizes images to [0,1] range.
    class VipsNormalize < Wrapper

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        image, y = @dataset[index]
        # Convert Vips image to pixel array and normalize to [0,1]
        [Numo::DFloat[image.to_a.flatten.map { |p| p / 255.0 }], y]
      end

    end

  end

end
