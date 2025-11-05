require 'ruby_neural_nets/transform_helpers/vips'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class VipsRemoveAlpha < ::Torch::NN::Module

        def forward(image)
          # Apply alpha channel removal to Vips image
          TransformHelpers::Vips.remove_alpha(image)
        end

      end
    end
  end
end
