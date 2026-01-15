require_relative '../../scenarios/data_loaders/shared/data_loader_scenarios'
require_relative '../../scenarios/data_loaders/shared/video_support_scenarios'
require "ruby_neural_nets/data_loaders/torch_images.#{RUBY_PLATFORM}"

describe RubyNeuralNets::DataLoaders::TorchImages do

  include_examples 'data loader scenarios',
    rotation_expected: [0.5, 0.7, 0, 0.05, 1, 0.05, 0, 0.7, 0.5],
    label_from: proc { |y| y.item },
    color_from: proc { |x| x.item }

  include_examples 'video support scenarios',
    label_from: proc { |y| y.item },
    color_from: proc { |x| x.item }

  # Creates a new data loader with default values for test scenarios.
  # Allows overriding specific default values through keyword arguments.
  #
  # Parameters::
  # * *overrides* (Hash): Keyword arguments to override default values
  # Result::
  # * RubyNeuralNets::DataLoader: The instantiated data loader
  def new_data_loader(**overrides)
    RubyNeuralNets::DataLoaders::TorchImages.new(
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
        flatten: true,
        video_slices_sec: 1.0,
        filter_dataset: 'all'
      }.merge(overrides)
    )
  end

  describe 'flatten option' do
        
    it 'does not flatten data when flatten is false' do
      with_test_dir(
        'test_dataset/class_0/test_image_0.png' => png(3, 2, [1000, 2000, 3000, 4000, 5000, 6000])
      ) do |datasets_path|
        x_tensor = new_data_loader(datasets_path: datasets_path, resize: [3, 2], flatten: false).dataset(:training).first.first.input
        expect(x_tensor.shape).to eq [1, 2, 3]
        expect_array_within(
          x_tensor.to_a,
          [
            [
              [0.015, 0.03, 0.045],
              [0.06, 0.075, 0.09]
            ]
          ]
        )
      end
    end

  end

end
