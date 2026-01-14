RSpec.shared_examples 'index filter scenarios' do |options|

  describe 'index filtering' do

    it 'filters correctly with individual indexes' do
      with_test_dir(
        'test_dataset/class_0/test_image_0.png' => png(1, 1, { color: [1] }),
        'test_dataset/class_0/test_image_1.png' => png(1, 1, { color: [2] }),
        'test_dataset/class_0/test_image_2.png' => png(1, 1, { color: [3] }),
        'test_dataset/class_0/test_image_3.png' => png(1, 1, { color: [4] }),
        'test_dataset/class_0/test_image_4.png' => png(1, 1, { color: [5] })
      ) do |datasets_path|
        data_loader = new_data_loader(datasets_path:, filter_dataset: '1,3', partitions: { training: 1.0, dev: 0.0, test: 0.0 }, minmax_normalize: false)
        training_samples = data_loader.dataset(:training).first.to_a
        expect(training_samples.size).to eq(2)
        # Check that we have samples with colors 2 and 4 (indexes 1 and 3)
        colors = training_samples.map { |sample| options[:color_from].call(sample.input).to_f }.sort
        expect(colors).to eq([2.0, 4.0])
      end
    end

    it 'filters correctly with ranges' do
      with_test_dir(
        'test_dataset/class_0/test_image_0.png' => png(1, 1, { color: [1] }),
        'test_dataset/class_0/test_image_1.png' => png(1, 1, { color: [2] }),
        'test_dataset/class_0/test_image_2.png' => png(1, 1, { color: [3] }),
        'test_dataset/class_0/test_image_3.png' => png(1, 1, { color: [4] }),
        'test_dataset/class_0/test_image_4.png' => png(1, 1, { color: [5] }),
        'test_dataset/class_0/test_image_5.png' => png(1, 1, { color: [6] })
      ) do |datasets_path|
        data_loader = new_data_loader(datasets_path:, filter_dataset: '1-3,5', partitions: { training: 1.0, dev: 0.0, test: 0.0 }, minmax_normalize: false)
        training_samples = data_loader.dataset(:training).first.to_a
        expect(training_samples.size).to eq(4)
        # Check that we have samples with colors 2,3,4,6 (indexes 1,2,3,5)
        colors = training_samples.map { |sample| options[:color_from].call(sample.input).to_f }.sort
        expect(colors).to eq([2.0, 3.0, 4.0, 6.0])
      end
    end

    it 'includes all indexes when filter is "all"' do
      with_test_dir(
        'test_dataset/class_0/test_image_0.png' => png(1, 1, { color: [1] }),
        'test_dataset/class_0/test_image_1.png' => png(1, 1, { color: [2] }),
        'test_dataset/class_0/test_image_2.png' => png(1, 1, { color: [3] })
      ) do |datasets_path|
        data_loader = new_data_loader(datasets_path:, filter_dataset: 'all', partitions: { training: 1.0, dev: 0.0, test: 0.0 }, minmax_normalize: false)
        training_samples = data_loader.dataset(:training).first.to_a
        expect(training_samples.size).to eq(3)
        # Check that we have all samples
        colors = training_samples.map { |sample| options[:color_from].call(sample.input).to_f }.sort
        expect(colors).to eq([1.0, 2.0, 3.0])
      end
    end

    it 'partitions correctly with filtered indexes' do
      with_test_dir(
        'test_dataset/class_0/test_image_0.png' => png(1, 1, { color: [1] }),
        'test_dataset/class_0/test_image_1.png' => png(1, 1, { color: [2] }),
        'test_dataset/class_0/test_image_2.png' => png(1, 1, { color: [3] }),
        'test_dataset/class_0/test_image_3.png' => png(1, 1, { color: [4] }),
        'test_dataset/class_0/test_image_4.png' => png(1, 1, { color: [5] })
      ) do |datasets_path|
        data_loader = new_data_loader(datasets_path:, filter_dataset: '0,2,4', partitions: { training: 0.6, dev: 0.4, test: 0.0 })
        # Should have 3 elements total (filtered), distributed as training: 2, dev: 1
        training_samples = data_loader.dataset(:training).first.to_a
        dev_samples = data_loader.dataset(:dev).first.to_a
        expect(training_samples.size).to eq(2)
        expect(dev_samples.size).to eq(1)
        # Check colors: original indexes 0,2,4 have colors 1,3,5
        all_colors = (training_samples + dev_samples).map { |sample| options[:color_from].call(sample.input).to_f }
        expect(all_colors.sort).to eq([1.0, 3.0, 5.0])
      end
    end

  end

end
