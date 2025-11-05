require 'ruby_neural_nets/transform_helpers/vips'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class VipsMinmaxNormalize < ::Torch::NN::Module

        def forward(image)
          # Apply min-max normalization to Vips image
          TransformHelpers::Vips.minmax_normalize(image)
        end

      end
    end
  end
end
