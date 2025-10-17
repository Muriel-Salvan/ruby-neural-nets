require 'torchvision'

module RubyNeuralNets
  
  module Torch
        
    module Transformers
      
      class RemoveAlpha < ::Torch::NN::Module

        def forward(tensor)
          tensor[..2, 0.., 0..]
        end

      end

      class Flatten < ::Torch::NN::Module

        def forward(tensor)
          # Reorder dimensions to make sure images are visible in the weights as well
          tensor.transpose(0, 1).transpose(1, 2).flatten
        end

      end

      class ToDouble < ::Torch::NN::Module

        def forward(tensor)
          # Make sure the resulting tensor is in 64 bits
          tensor.double
        end

      end

    end
    
    # Dataset that uses an underlying Dataset for a given dataset type
    class SampleFolderDataset < TorchVision::Datasets::DatasetFolder

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): The dataset containing files
      def initialize(dataset)
        @dataset = dataset
        super(
          @dataset.root,
          transform: TorchVision::Transforms::Compose.new([
            TorchVision::Transforms::ToTensor.new,
            Transformers::RemoveAlpha.new,
            Transformers::Flatten.new,
            Transformers::ToDouble.new,
          ])
        )
      end

      private
      
      def make_dataset(directory, class_to_idx, extensions, is_valid_file)
        @dataset.map { |(file, label)| [file, class_to_idx[label]] }
      end

    end

  end

end