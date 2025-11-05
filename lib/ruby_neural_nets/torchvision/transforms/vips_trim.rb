require 'ruby_neural_nets/transform_helpers/vips'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class VipsTrim < ::Torch::NN::Module

        def forward(image)
          # Apply trim transformation to Vips image
          TransformHelpers::Vips.trim(image)
        end

      end
    end
  end
end
