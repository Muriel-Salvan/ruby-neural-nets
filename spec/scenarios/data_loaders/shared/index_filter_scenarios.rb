RSpec.shared_examples 'index filter scenarios' do |options|

  describe 'index filtering' do

    it 'filters correctly with individual indexes' do
      with_test_dir(
        'test_dataset/class_0/test_image_0.png' => png(1, 1, { color: [257] }),
        'test_dataset/class_0/test_image_1.png' => png(1, 1, { color: [514] }),
        'test_dataset/class_0/test_image_2.png' => png(1, 1, { color: [771] }),
        'test_dataset/class_0/test_image_3.png' => png(1, 1, { color: [1028] }),
        'test_dataset/class_0/test_image_4.png' => png(1, 1, { color: [1285] })
      ) do |datasets_path|
        data_loader = new_data_loader(datasets_path:, filter_dataset: '1,3', partitions: { training: 1.0, dev: 0.0, test: 0.0 }, minmax_normalize: false)
        training_samples = data_loader.dataset(:training).first.to_a
        expect(training_samples.size).to eq(2)
        # Check that we have samples with colors 514 and 1028 (indexes 1 and 3)
        colors = training_samples.map { |sample| options[:color_from].call(sample.input).to_f }.sort
        expect(colors[0]).to be_within(1e-6).of(514.0/65535)
        expect(colors[1]).to be_within(1e-6).of(1028.0/65535)
      end
    end

    it 'filters correctly with ranges' do
      with_test_dir(
        'test_dataset/class_0/test_image_0.png' => png(1, 1, { color: [257] }),
        'test_dataset/class_0/test_image_1.png' => png(1, 1, { color: [514] }),
        'test_dataset/class_0/test_image_2.png' => png(1, 1, { color: [771] }),
        'test_dataset/class_0/test_image_3.png' => png(1, 1, { color: [1028] }),
        'test_dataset/class_0/test_image_4.png' => png(1, 1, { color: [1285] }),
        'test_dataset/class_0/test_image_5.png' => png(1, 1, { color: [1542] })
      ) do |datasets_path|
        data_loader = new_data_loader(datasets_path:, filter_dataset: '1-3,5', partitions: { training: 1.0, dev: 0.0, test: 0.0 }, minmax_normalize: false)
        training_samples = data_loader.dataset(:training).first.to_a
        expect(training_samples.size).to eq(4)
        # Check that we have samples with colors 514,771,1028,1542 (indexes 1,2,3,5)
        colors = training_samples.map { |sample| options[:color_from].call(sample.input).to_f }.sort
        expect(colors[0]).to be_within(1e-6).of(514.0/65535)
        expect(colors[1]).to be_within(1e-6).of(771.0/65535)
        expect(colors[2]).to be_within(1e-6).of(1028.0/65535)
        expect(colors[3]).to be_within(1e-6).of(1542.0/65535)
      end
    end

    it 'includes all indexes when filter is "all"' do
      with_test_dir(
        'test_dataset/class_0/test_image_0.png' => png(1, 1, { color: [257] }),
        'test_dataset/class_0/test_image_1.png' => png(1, 1, { color: [514] }),
        'test_dataset/class_0/test_image_2.png' => png(1, 1, { color: [771] })
      ) do |datasets_path|
        data_loader = new_data_loader(datasets_path:, filter_dataset: 'all', partitions: { training: 1.0, dev: 0.0, test: 0.0 }, minmax_normalize: false)
        training_samples = data_loader.dataset(:training).first.to_a
        expect(training_samples.size).to eq(3)
        # Check that we have all samples
        colors = training_samples.map { |sample| options[:color_from].call(sample.input).to_f }.sort
        expect(colors[0]).to be_within(1e-6).of(257.0/65535)
        expect(colors[1]).to be_within(1e-6).of(514.0/65535)
        expect(colors[2]).to be_within(1e-6).of(771.0/65535)
      end
    end

    it 'partitions correctly with filtered indexes' do
      with_test_dir(
        'test_dataset/class_0/test_image_0.png' => png(1, 1, { color: [257] }),
        'test_dataset/class_0/test_image_1.png' => png(1, 1, { color: [514] }),
        'test_dataset/class_0/test_image_2.png' => png(1, 1, { color: [771] }),
        'test_dataset/class_0/test_image_3.png' => png(1, 1, { color: [1028] }),
        'test_dataset/class_0/test_image_4.png' => png(1, 1, { color: [1285] })
      ) do |datasets_path|
        data_loader = new_data_loader(datasets_path:, filter_dataset: '0,2,4', partitions: { training: 0.6, dev: 0.4, test: 0.0 }, minmax_normalize: false)
        # Should have 3 elements total (filtered), distributed as training: 2, dev: 1
        training_samples = data_loader.dataset(:training).first.to_a
        dev_samples = data_loader.dataset(:dev).first.to_a
        expect(training_samples.size).to eq(2)
        expect(dev_samples.size).to eq(1)
        # Check colors: original indexes 0,2,4 have colors 257,771,1285
        all_colors = (training_samples + dev_samples).map { |sample| options[:color_from].call(sample.input).to_f }.sort
        expect(all_colors[0]).to be_within(1e-6).of(257.0/65535)
        expect(all_colors[1]).to be_within(1e-6).of(771.0/65535)
        expect(all_colors[2]).to be_within(1e-6).of(1285.0/65535)
      end
    end

  end

end
