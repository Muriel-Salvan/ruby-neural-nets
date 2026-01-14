require_relative 'shared/data_loader_scenarios'
require_relative 'shared/video_support_scenarios'
require 'ruby_neural_nets/data_loaders/numo_image_magick'

describe RubyNeuralNets::DataLoaders::NumoImageMagick do

  include_examples 'data loader scenarios with index filtering',
    rotation_expected: [0.5, 0.7, 0, 0.05, 1, 0.05, 0, 0.7, 0.5],
    label_from: proc { |y| y.max_index },
    color_from: proc { |x| x }

  include_examples 'video support scenarios',
    label_from: proc { |y| y.max_index },
    color_from: proc { |x| x }

  # Creates a new NumoImageMagick data loader with default values for test scenarios.
  # Allows overriding specific default values through keyword arguments.
  #
  # Parameters::
  # * *overrides* (Hash): Keyword arguments to override default values
  # Result::
  # * RubyNeuralNets::DataLoaders::NumoImageMagick: The instantiated data loader
  def new_data_loader(**overrides)
    RubyNeuralNets::DataLoaders::NumoImageMagick.new(
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
        minmax_normalize: false,
        video_slices_sec: 1.0,
        filter_dataset: 'all'
      }.merge(overrides)
    )
  end

end
