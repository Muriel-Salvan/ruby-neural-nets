require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/transform_helpers/vips'
require 'ruby_neural_nets/sample'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies image resizing.
    class VipsResize < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be wrapped
      # * *resize* (Array): Array of 2 integers [width, height] for resizing
      def initialize(dataset, resize:)
        super(dataset)
        @target_width, @target_height = resize
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
          -> { TransformHelpers::Vips.resize(sample.input, @target_width, @target_height) },
          -> { sample.target }
        )
      end

      # Get some images stats.
      # Those are supposed to be the same for all samples from the dataset and can be used to compute the model's architecture.
      #
      # Result::
      # * Hash: Image stats:
      #   * *rows* (Integer or nil): Number of rows if it applies to all images, or nil otherwise
      #   * *cols* (Integer or nil): Number of columns if it applies to all images, or nil otherwise
      #   * *channels* (Integer or nil): Number of channels if it applies to all images, or nil otherwise
      #   * *depth* (Integer or nil): Depth (number of bits) used to encode pixel channel's values if it applies to all images, or nil otherwise
      def image_stats
        @dataset.image_stats.merge(
          rows: @target_height,
          cols: @target_width
        )
      end

    end

  end

end
