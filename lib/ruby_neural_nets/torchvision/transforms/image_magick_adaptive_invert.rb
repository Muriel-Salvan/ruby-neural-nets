require 'ruby_neural_nets/transform_helpers/image_magick'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class ImageMagickAdaptiveInvert < ::Torch::NN::Module

        def forward(image)
          # Apply adaptive invert transformation to ImageMagick image
          TransformHelpers::ImageMagick.adaptive_invert(image)
        end

      end
    end
  end
end
