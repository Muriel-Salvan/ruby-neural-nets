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

    end
    
    # Dataset that uses an underlying Dataset for a given dataset type
    class SampleFolderDataset < TorchVision::Datasets::DatasetFolder

      # Constructor
      #
      # Parameters::
      # * *dataset* (Dataset): The dataset containing files
      # * *dataset_type* (Symbol): The dataset type
      def initialize(dataset, dataset_type)
        @dataset = dataset
        @dataset_type = dataset_type
        super(
          @dataset.root,
          transform: TorchVision::Transforms::Compose.new([
            TorchVision::Transforms::ToTensor.new,
            Transformers::RemoveAlpha.new,
            Transformers::Flatten.new
          ])
        )
      end

      private
      
      def make_dataset(directory, class_to_idx, extensions, is_valid_file)
        @dataset.classes.map do |class_name|
          class_idx = class_to_idx[class_name]
          @dataset.files(@dataset_type, class_name).map { |file| [file, class_idx] }
        end.flatten(1)
      end

    end

  end

end