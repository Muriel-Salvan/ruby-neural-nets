require 'ruby_neural_nets/data_loaders/numo_image_magick'

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

  it 'returns the right image stats for grayscale images' do
    with_test_fs('datasets/test_dataset/class_0/test_image_0.png' => png(10, 20, { color: [128] })) do
      expect(new_data_loader(resize: [10, 20]).image_stats).to eq({ rows: 20, cols: 10, channels: 1 })
    end
  end

  it 'returns the right image stats for color images' do
    with_test_fs('datasets/test_dataset/class_0/test_image_0.png' => png(10, 20, { color: [255, 128, 64] })) do
      expect(new_data_loader(resize: [10, 20]).image_stats).to eq({ rows: 20, cols: 10, channels: 3 })
    end
  end

  it 'returns the right labels' do
    with_test_fs(
      'datasets/test_dataset/class_a/test_image_0.png' => png(1, 1, { color: [0] }),
      'datasets/test_dataset/class_b/test_image_0.png' => png(1, 1, { color: [128] }),
      'datasets/test_dataset/class_c/test_image_0.png' => png(1, 1, { color: [255] })
    ) do
      expect(new_data_loader.labels.sort).to eq(['class_a', 'class_b', 'class_c'])
    end
  end

  it 'returns the right labels statistics' do
    with_test_fs(
      'datasets/test_dataset/class_a/test_image_0.png' => png(1, 1, { color: [0] }),
      'datasets/test_dataset/class_a/test_image_1.png' => png(1, 1, { color: [0] }),
      'datasets/test_dataset/class_b/test_image_0.png' => png(1, 1, { color: [128] }),
      'datasets/test_dataset/class_c/test_image_0.png' => png(1, 1, { color: [255] }),
      'datasets/test_dataset/class_c/test_image_1.png' => png(1, 1, { color: [255] }),
      'datasets/test_dataset/class_c/test_image_2.png' => png(1, 1, { color: [255] })
    ) do
      data_loader = new_data_loader(partitions: { training: 0.5, dev: 0.3, test: 0.2 })
      expect(data_loader.label_stats(:training)).to eq({
        'class_a' => { nbr_elements: 1 },
        'class_b' => { nbr_elements: 1 },
        'class_c' => { nbr_elements: 2 }
      })
      expect(data_loader.label_stats(:dev)).to eq({
        'class_a' => { nbr_elements: 1 },
        'class_b' => { nbr_elements: 0 },
        'class_c' => { nbr_elements: 1 }
      })
      expect(data_loader.label_stats(:test)).to eq({
        'class_a' => { nbr_elements: 0 },
        'class_b' => { nbr_elements: 0 },
        'class_c' => { nbr_elements: 0 }
      })
    end
  end

  it 'partitions correctly the dataset' do
    with_test_fs(
      # 3 classes, having the following number of files:
      # 0: 3 + 2 + 1 = 6
      # 1: 6 + 4 + 2 = 12
      # 2: 9 + 6 + 3 = 18
      (0..2).map do |class_idx|
        ((class_idx + 1) * 6).times.map do |img_idx|
          [
            "datasets/test_dataset/class_#{class_idx}/test_image_#{img_idx}.png",
            png(1, 1, { color: [class_idx * 18 + img_idx] })
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

  it 'applies nbr_clones correctly' do
    with_test_fs(
      'datasets/test_dataset/class_0/test_image_0.png' => png(1, 1, { color: [32768] })
    ) do
      elements = new_data_loader(nbr_clones: 3).dataset(:training).first.each_element.to_a
      expect(elements.size).to eq(3)
      # Check that each minibatch element is the same
      expect(elements.map { |x, y| [x.to_a, y.to_a] }.uniq.size).to eq(1)
    end
  end

  it 'applies rot_angle correctly' do
    with_test_fs(
      'datasets/test_dataset/class_0/test_image_0.png' => png(3, 3, [0, 65535, 0, 0, 65535, 0, 0, 65535, 0])
    ) do
      # Since rotation is random within the angle, we check that the vertical bar is rotated a bit
      expect_array_within(new_data_loader(rot_angle: 90.0, resize: [3, 3]).dataset(:training).first.each_element.first[0], [0.5, 0.7, 0, 0.05, 1, 0.05, 0, 0.7, 0.5])
    end
  end

  it 'applies adaptive_invert correctly' do
    with_test_fs(
      'datasets/test_dataset/class_0/test_image_0.png' => png(1, 1, { color: [0] })  # Black pixel, should invert to white
    ) do
      # After adaptive invert, the black pixel should become close to white
      expect(new_data_loader(adaptive_invert: true).dataset(:training).first.each_element.first[0].to_a[0]).to be > 0.9
    end
  end

  it 'applies trim correctly' do
    # Create an image with a white border (255) and black center (0) that can be trimmed
    with_test_fs(
      'datasets/test_dataset/class_0/test_image_0.png' => png(4, 4, [65535, 65535, 65535, 65535, 65535, 0, 0, 65535, 65535, 0, 0, 65535, 65535, 65535, 65535, 65535])
    ) do 
      # After trim and resize, the colors should be close to the inner rectangle (0)
      expect(new_data_loader(trim: true, resize: [4, 4]).dataset(:training).first.each_element.first[0].to_a).to eq([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    end
  end

  it 'applies noise_intensity correctly' do
    with_test_fs(
      'datasets/test_dataset/class_0/test_image_0.png' => png(3, 3, { color: [32768] })
    ) do
      x = new_data_loader(resize: [3, 3], noise_intensity: 0.1).dataset(:training).first.each_element.first[0].to_a
      # With noise, the pixel value should be different from the original, different between themselves but with the same mean
      expect_array_not_within(x, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
      expect(x.sum / x.size).to be_within(0.1).of(0.5)
      expect(x.max - x.min).to be_within(0.1).of(0.5)
    end
  end

  it 'applies minmax_normalize correctly on grayscale images' do
    with_test_fs(
      'datasets/test_dataset/class_0/test_image_0.png' => png(2, 2, [100, 200, 300, 400])
    ) do
      expect_array_within(new_data_loader(minmax_normalize: true, resize: [2, 2]).dataset(:training).first.each_element.first[0], [0, 0.33, 0.66, 1])
    end
  end

  it 'applies minmax_normalize correctly on color images per channel' do
    with_test_fs(
      'datasets/test_dataset/class_0/test_image_0.png' => png(2, 2, [10, 100, 1000, 20, 200, 2000, 30, 300, 3000, 40, 400, 4000])
    ) do
      expect_array_within(new_data_loader(minmax_normalize: true, resize: [2, 2]).dataset(:training).first.each_element.first[0], [0, 0, 0, 0.33, 0.33, 0.33, 0.66, 0.66, 0.66, 1, 1, 1])
    end
  end

end
