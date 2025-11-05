require 'ruby_neural_nets/transform_helpers/vips'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class VipsResize < ::Torch::NN::Module

        # Get the size for resize
        #   [width, height]
        attr_reader :size

        def initialize(size)
          @size = size
        end

        def forward(image)
          # Apply resize transformation to Vips image using our helper
          TransformHelpers::Vips.resize(image, *@size)
        end

      end
    end
  end
end
