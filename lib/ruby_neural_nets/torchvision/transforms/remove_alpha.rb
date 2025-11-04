module RubyNeuralNets
  module TorchVision
    module Transforms
      class RemoveAlpha < ::Torch::NN::Module

        def forward(tensor)
          tensor[..2, 0.., 0..]
        end

      end
    end
  end
end
