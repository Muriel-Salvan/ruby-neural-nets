require 'ruby_neural_nets/transform_helpers'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class TrimImagemagick < ::Torch::NN::Module

        def forward(image)
          # Apply trim transformation to ImageMagick image
          TransformHelpers.trim(image)
        end

      end
    end
  end
end
