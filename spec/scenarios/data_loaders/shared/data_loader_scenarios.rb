RSpec.shared_examples 'data loader scenarios' do |options|

  describe 'image stats' do
    it 'returns the right image stats for grayscale images' do
      # Don't use fully black images as they are encoded as 8 bits by ImageMagick automatically
      with_test_fs('datasets/test_dataset/class_0/test_image_0.png' => png(10, 20, { color: [1] })) do
        expect(new_data_loader(resize: [10, 20]).image_stats).to eq({ rows: 20, cols: 10, channels: 1, depth: 16 })
      end
    end

    it 'returns the right image stats for color images' do
      with_test_fs('datasets/test_dataset/class_0/test_image_0.png' => png(10, 20, { color: [0, 10, 20] })) do
        expect(new_data_loader(resize: [10, 20]).image_stats).to eq({ rows: 20, cols: 10, channels: 3, depth: 16 })
      end
    end
  end

  describe 'labels' do
    it 'returns the right labels' do
      with_test_fs(
        'datasets/test_dataset/class_a/test_image_0.png' => png(1, 1, { color: [1] }),
        'datasets/test_dataset/class_b/test_image_0.png' => png(1, 1, { color: [1] }),
        'datasets/test_dataset/class_c/test_image_0.png' => png(1, 1, { color: [1] })
      ) do
        expect(new_data_loader.labels.sort).to eq(['class_a', 'class_b', 'class_c'])
      end
    end

    it 'returns the right labels statistics' do
      with_test_fs(
        'datasets/test_dataset/class_a/test_image_0.png' => png(1, 1, { color: [1] }),
        'datasets/test_dataset/class_a/test_image_1.png' => png(1, 1, { color: [1] }),
        'datasets/test_dataset/class_b/test_image_0.png' => png(1, 1, { color: [1] }),
        'datasets/test_dataset/class_c/test_image_0.png' => png(1, 1, { color: [1] }),
        'datasets/test_dataset/class_c/test_image_1.png' => png(1, 1, { color: [1] }),
        'datasets/test_dataset/class_c/test_image_2.png' => png(1, 1, { color: [1] })
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
  end

  describe 'partitioning' do
    it 'partitions correctly the dataset' do
      with_test_fs(
        # 3 classes, having the following number of files:
        #    training   dev   test
        # 0: 3        + 2   + 1    = 6
        # 1: 6        + 4   + 2    = 12
        # 2: 9        + 6   + 3    = 18
        # Each file has a different color.
        (0..2).map do |class_idx|
          ((class_idx + 1) * 6).times.map do |img_idx|
            [
              "datasets/test_dataset/class_#{class_idx}/test_image_#{img_idx}.png",
              png(1, 1, { color: [class_idx * 18 + img_idx + 1] })
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

  describe 'cloning' do
    it 'applies nbr_clones correctly' do
      with_test_fs(
        'datasets/test_dataset/class_0/test_image_0.png' => png(1, 1, { color: [1] })
      ) do
        elements = new_data_loader(nbr_clones: 3).dataset(:training).first.each_element.to_a
        expect(elements.size).to eq(3)
        # Check that each minibatch element is the same
        expect_array_within(
          elements.map { |x, y| [x.to_a, y.to_a] },
          [
            [[0], [1]],
            [[0], [1]],
            [[0], [1]]
          ]
        )
      end
    end
  end

  describe 'rotation' do
    it 'applies rot_angle correctly' do
      with_test_fs(
        'datasets/test_dataset/class_0/test_image_0.png' => png(3, 3, [0, 65535, 0, 0, 65535, 0, 0, 65535, 0])
      ) do
        # Since rotation is random within the angle, we check that the vertical bar is rotated a bit
        expect_array_within(new_data_loader(rot_angle: 90.0, resize: [3, 3]).dataset(:training).first.each_element.first[0], options[:rotation_expected])
      end
    end
  end

  describe 'adaptive invert' do
    it 'applies adaptive_invert correctly on black images' do
      with_test_fs(
        'datasets/test_dataset/class_0/test_image_0.png' => png(1, 1, { color: [1] })
      ) do
        # Black pixel, should invert to white
        expect_array_within(new_data_loader(adaptive_invert: true).dataset(:training).first.each_element.first[0].to_a, [1])
      end
    end

    it 'applies adaptive_invert correctly on white images' do
      with_test_fs(
        'datasets/test_dataset/class_0/test_image_0.png' => png(1, 1, { color: [65535] })
      ) do
        # White pixel, should stay white
        expect_array_within(new_data_loader(adaptive_invert: true).dataset(:training).first.each_element.first[0].to_a, [1])
      end
    end
  end

  describe 'trimming' do
    it 'applies trim correctly' do
      # Create an image with a white border and black center (0) that can be trimmed
      # Vips does not detect correctly borders on 2x2 zones.
      # Therefore we use 3x3 zone detected in a 5x5 image.
      with_test_fs(
        'datasets/test_dataset/class_0/test_image_0.png' => png(5, 5, Array.new(25, 65535).fill(0, 6, 3).fill(0, 11, 3).fill(0, 16, 3))
      ) do
        # After trim and resize, the colors should be close to the inner rectangle (0)
        expect(new_data_loader(trim: true, resize: [5, 5]).dataset(:training).first.each_element.first[0].to_a).to eq([0] * 25)
      end
    end
  end

  describe 'noise' do
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
  end

  describe 'minmax normalization' do
    it 'applies minmax_normalize correctly on grayscale images' do
      with_test_fs(
        'datasets/test_dataset/class_0/test_image_0.png' => png(2, 2, [10, 20, 30, 40])
      ) do
        expect_array_within(new_data_loader(minmax_normalize: true, resize: [2, 2]).dataset(:training).first.each_element.first[0], [0, 0.33, 0.66, 1])
      end
    end

    it 'applies minmax_normalize correctly on color images' do
      with_test_fs(
        'datasets/test_dataset/class_0/test_image_0.png' => png(2, 2, [
          # R, G, B
          10, 50, 40,
          20, 50, 30,
          30, 40, 20,
          40, 40, 10
        ])
      ) do
        expect_array_within(new_data_loader(minmax_normalize: true, resize: [2, 2]).dataset(:training).first.each_element.first[0], [
          # R,  G, B
          0,    1, 1,
          0.33, 1, 0.66,
          0.66, 0, 0.33,
          1,    0, 0
        ])
      end
    end
  end
end
