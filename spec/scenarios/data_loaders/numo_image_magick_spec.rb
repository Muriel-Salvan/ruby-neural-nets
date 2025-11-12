require 'ruby_neural_nets/data_loaders/numo_image_magick'

require 'ruby_neural_nets_test/helpers'

describe RubyNeuralNets::DataLoaders::NumoImageMagick do
  
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

  it 'partitions correctly the dataset' do
    RubyNeuralNetsTest::Helpers.with_test_fs(
      # 3 classes, having the following number of files:
      # 0: 3 + 2 + 1 = 6
      # 1: 6 + 4 + 2 = 12
      # 2: 9 + 6 + 3 = 18
      (0..2).map do |class_idx|
        ((class_idx + 1) * 6).times.map do |img_idx|
          [
            "datasets/test_dataset/class_#{class_idx}/test_image_#{img_idx}.png",
            RubyNeuralNetsTest::Helpers.generate_png(1, 1, [class_idx * 18 + img_idx])
          ]
        end
      end.flatten(1).to_h
    ) do
      data_loader = new_data_loader(partitions: { training: 0.5, dev: 0.33, test: 0.17 })
      # For each dataset type, validate the minibatches by checking number of unique pixel colors for each class
      expect(
        [:training, :dev, :test].to_h do |dataset_type|
          [
            dataset_type,
            data_loader.dataset(dataset_type).
              map { |minibatch| minibatch.each_element.map { |x, y| [x[0], y.max_index] } }.
              flatten(1).
              group_by { |(_color, class_idx)| class_idx }.
              to_h do |class_idx, elements|
                [
                  class_idx,
                  elements.map { |(color, _class_idx)| color }.uniq.size
                ]
              end
          ]
        end
      ).to eq(
        {
          training: {
            0 => 3,
            1 => 6,
            2 => 9
          },
          dev: {
            0 => 2,
            1 => 4,
            2 => 6
          },
          test: {
            0 => 1,
            1 => 2,
            2 => 3
          }
        }
      )
    end
  end

end
