require_relative '../../scenarios/data_loaders/shared/data_loader_scenarios'
require "ruby_neural_nets/data_loaders/torch_image_magick.#{RUBY_PLATFORM}"

describe RubyNeuralNets::DataLoaders::TorchImageMagick do
  include_examples 'data loader scenarios', 
    rotation_expected: [0.5, 0.7, 0, 0.05, 1, 0.05, 0, 0.7, 0.5],
    labels_as_onehot: false

  # Creates a new data loader with default values for test scenarios.
  # Allows overriding specific default values through keyword arguments.
  #
  # Parameters::
  # * *overrides* (Hash): Keyword arguments to override default values
  # Result::
  # * RubyNeuralNets::DataLoader: The instantiated data loader
  def new_data_loader(**overrides)
    RubyNeuralNets::DataLoaders::TorchImageMagick.new(
      **{
        datasets_path: './datasets',
        dataset: 'test_dataset',
        max_minibatch_size: 10,
        dataset_seed: 42,
        partitions: { training: 0.7, dev: 0.15, test: 0.15 },
        nbr_clones: 1,
        rot_angle: 0.0,
        grayscale: false,
        adaptive_invert: false,
        trim: false,
        resize: [1, 1],
        noise_intensity: 0.0,
        minmax_normalize: false
      }.merge(overrides)
    )
  end
end
