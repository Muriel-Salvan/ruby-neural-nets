require 'ruby_neural_nets/transform_helpers/image_magick'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class ImageMagickTrim < ::Torch::NN::Module

        def forward(image)
          # Apply trim transformation to ImageMagick image
          TransformHelpers::ImageMagick.trim(image)
        end

      end
    end
  end
end
