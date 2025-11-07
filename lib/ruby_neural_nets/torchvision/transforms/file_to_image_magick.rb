module RubyNeuralNets
  module TorchVision
    module Transforms
      class FileToImageMagick < ::Torch::NN::Module

        def forward(file_path)
          # Load ImageMagick image from file path
          Magick::Image.read(file_path).first
        end

      end
    end
  end
end
