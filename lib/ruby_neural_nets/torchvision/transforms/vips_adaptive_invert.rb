require 'ruby_neural_nets/transform_helpers/vips'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class VipsAdaptiveInvert < ::Torch::NN::Module

        def forward(image)
          # Apply adaptive invert transformation to Vips image
          TransformHelpers::Vips.adaptive_invert(image)
        end

      end
    end
  end
end
