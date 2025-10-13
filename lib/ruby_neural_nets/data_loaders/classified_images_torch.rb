require 'ruby_neural_nets/data_loaders/classified_images'
require 'ruby_neural_nets/torch/sample_folder_dataset'

module RubyNeuralNets

  module DataLoaders

    # Dataset of classified images, to be used by Torch.rb
    class ClassifiedImagesTorch < ClassifiedImages

      # Loop over the dataset using minibatches
      #
      # Parameters::
      # * *dataset_type* (Symbol): Dataset type to loop over (can be :train, :dev or :test)
      # * *max_minibatch_size* (Integer): Max size each minibatch should have
      # * Code: Code called for each minibatch
      #   * *minibatch_x* (Object): Read minibatch X
      #   * *minibatch_y* (Object): Read minibatch Y
      #   * *minibatch_size* (Integer): Minibatches size
      def for_each_minibatch(dataset_type, max_minibatch_size)
        ::Torch::Utils::Data::DataLoader.new(RubyNeuralNets::Torch::SampleFolderDataset.new(self, dataset_type), batch_size: max_minibatch_size, shuffle: true).each do |inputs, labels|
          yield inputs, labels, inputs.shape[0]
        end
      end

    end

  end

end
