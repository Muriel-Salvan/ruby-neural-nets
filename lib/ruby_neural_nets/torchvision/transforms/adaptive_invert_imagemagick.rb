require 'ruby_neural_nets/transform_helpers'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class AdaptiveInvertImagemagick < ::Torch::NN::Module

        def forward(image)
          # Apply adaptive invert transformation to ImageMagick image
          TransformHelpers.adaptive_invert(image)
        end

      end
    end
  end
end
