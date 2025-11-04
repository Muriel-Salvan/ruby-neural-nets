require 'ruby_neural_nets/transform_helpers/image_magick'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class ImageMagickGrayscale < ::Torch::NN::Module

        def forward(image)
          # Apply grayscale transformation to ImageMagick image
          TransformHelpers::ImageMagick.grayscale(image)
        end

      end
    end
  end
end
