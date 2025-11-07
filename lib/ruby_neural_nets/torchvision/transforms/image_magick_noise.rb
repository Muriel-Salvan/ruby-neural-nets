require 'ruby_neural_nets/transform_helpers/image_magick'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class ImageMagickNoise < ::Torch::NN::Module

        # Get the noise intensity
        attr_reader :noise_intensity

        def initialize(noise_intensity, numo_rng)
          @noise_intensity = noise_intensity
          @numo_rng = numo_rng
        end

        def forward(image)
          # Apply Gaussian noise transformation to ImageMagick image
          TransformHelpers::ImageMagick.gaussian_noise(image, @noise_intensity, @numo_rng)
        end

      end
    end
  end
end
