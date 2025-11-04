require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/transform_helpers'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies random image rotation.
    class ImageRotate < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be wrapped
      # * *rng* (Random): Random number generator for rotations
      # * *rot_angle* (Float): Maximum rotation angle in degrees (rotation will be between -rot_angle and +rot_angle)
      def initialize(dataset, rng:, rot_angle:)
        super(dataset)
        @rng = rng
        @rot_angle = rot_angle
      end

      # Access an element of the dataset
      #
      # Parameters::
      # * *index* (Integer): Index of the dataset element to access
      # Result::
      # * Object: The element X of the dataset
      # * Object: The element Y of the dataset
      def [](index)
        image, y = @dataset[index]
        [TransformHelpers.rotate(image, @rot_angle, @rng), y]
      end

    end

  end

end
