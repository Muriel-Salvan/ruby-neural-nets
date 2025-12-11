require_relative 'shared/data_loader_scenarios'
require 'ruby_neural_nets/data_loaders/numo_vips'

describe RubyNeuralNets::DataLoaders::NumoVips do
  include_examples 'data loader scenarios',
    rotation_expected: [0.03, 0.32, 0, 0, 0.92, 0.15, 0, 0.54, 0.54]

  # Creates a new NumoVips data loader with default values for test scenarios.
  # Allows overriding specific default values through keyword arguments.
  #
  # Parameters::
  # * *overrides* (Hash): Keyword arguments to override default values
  # Result::
  # * RubyNeuralNets::DataLoaders::NumoVips: The instantiated data loader
  def new_data_loader(**overrides)
    RubyNeuralNets::DataLoaders::NumoVips.new(
      **{
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
