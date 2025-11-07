require 'ruby_neural_nets/transform_helpers/image_magick'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class ImageMagickRemoveAlpha < ::Torch::NN::Module

        def forward(image)
          # Apply alpha channel removal to ImageMagick image
          TransformHelpers::ImageMagick.remove_alpha(image)
        end

      end
    end
  end
end
