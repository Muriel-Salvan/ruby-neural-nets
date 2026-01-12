require 'ruby_neural_nets/helpers'
require 'ruby_neural_nets/datasets/wrapper'
require 'ruby_neural_nets/transform_helpers/vips'
require 'ruby_neural_nets/sample'
require 'numo/narray'

module RubyNeuralNets

  module Datasets

    # Dataset wrapper that applies Gaussian noise to images.
    class VipsNoise < Wrapper

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): Dataset to be wrapped
      # * *numo_rng* (Numo::Random::Generator): Random number generator for Numo noise
      # * *noise_intensity* (Float): Intensity of Gaussian noise for transformations
      def initialize(dataset, numo_rng:, noise_intensity:)
        super(dataset)
        @numo_rng = numo_rng
        @noise_intensity = noise_intensity
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
          -> { TransformHelpers::Vips.gaussian_noise(sample.input, @noise_intensity, @numo_rng) },
          -> { sample.target }
        )
      end

    end

  end

end
