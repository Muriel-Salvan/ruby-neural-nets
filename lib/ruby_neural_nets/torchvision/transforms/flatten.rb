module RubyNeuralNets
  module TorchVision
    module Transforms
      class Flatten < ::Torch::NN::Module

        def forward(tensor)
          # Reorder dimensions to make sure images are visible in the weights as well
          tensor.transpose(0, 1).transpose(1, 2).flatten
        end

      end
    end
  end
end
