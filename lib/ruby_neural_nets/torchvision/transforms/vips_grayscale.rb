require 'ruby_neural_nets/transform_helpers/vips'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class VipsGrayscale < ::Torch::NN::Module

        def forward(image)
          # Apply grayscale transformation to Vips image
          TransformHelpers::Vips.grayscale(image)
        end

      end
    end
  end
end
