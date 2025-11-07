require 'ruby_neural_nets/transform_helpers/image_magick'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class ImageMagickRotate < ::Torch::NN::Module

        # Get the maximum rotation angle
        attr_reader :rot_angle

        def initialize(rot_angle, rng)
          @rot_angle = rot_angle
          @rng = rng
        end

        def forward(image)
          # Apply random rotation transformation to ImageMagick image
          TransformHelpers::ImageMagick.rotate(image, @rot_angle, @rng)
        end

      end
    end
  end
end
