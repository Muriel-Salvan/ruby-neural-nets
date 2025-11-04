require 'ruby_neural_nets/transform_helpers/image_magick'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class ImageMagickResize < ::Torch::NN::Module

        # Get the size for resize
        #   [width, height]
        attr_reader :size

        def initialize(size)
          @size = size
        end

        def forward(image)
          # Apply resize transformation to ImageMagick image
          TransformHelpers::ImageMagick.resize(image, *@size)
        end

      end
    end
  end
end
