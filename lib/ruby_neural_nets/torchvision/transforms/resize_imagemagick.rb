require 'ruby_neural_nets/transform_helpers'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class ResizeImagemagick < ::Torch::NN::Module

        # Get the size for resize
        #   [width, height]
        attr_reader :size

        def initialize(size)
          @size = size
        end

        def forward(image)
          # Apply resize transformation to ImageMagick image
          TransformHelpers.resize(image, *@size)
        end

      end
    end
  end
end
