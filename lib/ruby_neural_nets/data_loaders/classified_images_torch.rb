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
        @data_loader_cache = {} unless defined?(@data_loader_cache)
        @data_loader_cache[max_minibatch_size] = {} unless @data_loader_cache.key?(max_minibatch_size)
        @data_loader_cache[max_minibatch_size][dataset_type] = ::Torch::Utils::Data::DataLoader.new(RubyNeuralNets::Torch::SampleFolderDataset.new(self, dataset_type), batch_size: max_minibatch_size, shuffle: true).each.to_a unless @data_loader_cache[max_minibatch_size].key?(dataset_type)
        @data_loader_cache[max_minibatch_size][dataset_type].shuffle.each do |(inputs, labels)|
          yield inputs, labels, inputs.shape[0]
        end
      end

    end

  end

end
