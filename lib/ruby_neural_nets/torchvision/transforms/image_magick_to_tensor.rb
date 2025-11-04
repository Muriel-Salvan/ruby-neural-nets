require 'ruby_neural_nets/helpers'

module RubyNeuralNets
  module TorchVision
    module Transforms
      class ImageMagickToTensor < ::Torch::NN::Module

        def forward(image)
          # Convert ImageMagick image to Torch tensor
          pixels_map = Helpers.image_pixels_map(image)
          # Convert to Numo array first, then to Torch tensor
          # Reshape based on number of channels
          # Convert to CHW format and then to Torch tensor
          ::Torch.from_numo(
            Numo::DFloat[image.export_pixels(0, 0, image.columns, image.rows, pixels_map)].
              reshape(image.rows, image.columns, pixels_map == 'I' ? 1 : 3).
              transpose(2, 0, 1)
          ).div(65535)
        end

      end
    end
  end
end
