require 'ruby_neural_nets/transform_helpers/vips'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class Vips8Bits < ::Torch::NN::Module

        def forward(image)
          # Convert Vips image to 8-bit (uchar) format
          TransformHelpers::Vips.to_8_bits(image)
        end

      end
    end
  end
end
