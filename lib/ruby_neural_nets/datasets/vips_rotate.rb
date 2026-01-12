require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/transform_helpers/vips'
require 'ruby_neural_nets/sample'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies random image rotation.
    class VipsRotate < Wrapper

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
      # * Sample: The sample containing input and target data
      def [](index)
        sample = @dataset[index]
        Sample.new(
          -> { TransformHelpers::Vips.rotate(sample.input, @rot_angle, @rng) },
          -> { sample.target }
        )
      end

    end

  end

end
