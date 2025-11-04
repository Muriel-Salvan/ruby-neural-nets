require 'ruby_neural_nets/transform_helpers/image_magick'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class ImageMagickMinmaxNormalize < ::Torch::NN::Module

        def forward(image)
          # Apply min-max normalization to ImageMagick image
          TransformHelpers::ImageMagick.minmax_normalize(image)
        end

      end
    end
  end
end
