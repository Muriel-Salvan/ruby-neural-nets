module RubyNeuralNets
  module TorchVision
    module Transforms
      class ToDouble < ::Torch::NN::Module

        def forward(tensor)
          # Make sure the resulting tensor is in 64 bits
          tensor.double
        end

      end
    end
  end
end
