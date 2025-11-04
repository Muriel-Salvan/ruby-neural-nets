require 'ruby_neural_nets/transform_helpers'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class GrayscaleImagemagick < ::Torch::NN::Module

        def forward(image)
          # Apply grayscale transformation to ImageMagick image
          TransformHelpers.grayscale(image)
        end

      end
    end
  end
end
