require 'ruby_neural_nets/transform_helpers/vips'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class VipsNoise < ::Torch::NN::Module

        # Get the noise intensity
        attr_reader :noise_intensity

        def initialize(noise_intensity, numo_rng)
          @noise_intensity = noise_intensity
          @numo_rng = numo_rng
        end

        def forward(image)
          # Apply Gaussian noise transformation to Vips image
          TransformHelpers::Vips.gaussian_noise(image, @noise_intensity, @numo_rng)
        end

      end
    end
  end
end
