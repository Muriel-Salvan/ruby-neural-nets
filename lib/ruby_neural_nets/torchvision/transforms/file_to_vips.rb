module RubyNeuralNets
  module TorchVision
    module Transforms
      class FileToVips < ::Torch::NN::Module

        def forward(file_path)
          # Load Vips image from file path
          ::Vips::Image.new_from_file(file_path)
        end

      end
    end
  end
end
