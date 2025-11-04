require 'ruby_neural_nets/transform_helpers'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class MinmaxNormalizeImagemagick < ::Torch::NN::Module

        def forward(image)
          # Apply min-max normalization to ImageMagick image
          TransformHelpers.minmax_normalize(image)
        end

      end
    end
  end
end
