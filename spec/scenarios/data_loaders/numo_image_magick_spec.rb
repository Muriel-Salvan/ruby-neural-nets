require 'ruby_neural_nets/data_loaders/numo_image_magick'

require 'ruby_neural_nets_test/helpers'

describe RubyNeuralNets::DataLoaders::NumoImageMagick do

  describe 'minibatch data validation' do
    it 'serves correct minibatch X and Y for various datasets with simple dataset' do
      # Define the mocked filesystem with 10 files per class
      files = {}
      (0..2).each do |class_idx|
        (0..9).each do |img_idx|
          file_path = "datasets/test_simple_dataset/#{class_idx}/test_image_#{img_idx}.png"

          # Create a real Magick::Image with test data
          mock_image = Magick::Image.new(28, 28) do |img|
            img.format = 'PNG'
          end

          # Generate deterministic pixel data based on the class
          pixel_data = Array.new(28 * 28) do |j|
            case class_idx
            when 0
              0
            when 1
              100
            when 2
              200
            end
          end

          # Import the pixel data into the image
          mock_image.import_pixels(0, 0, 28, 28, 'I', pixel_data)

          # Store the PNG data
          files[file_path] = mock_image.to_blob
        end
      end

      RubyNeuralNetsTest::Helpers.with_test_fs(files) do
        # Create data loader inside fakefs
        data_loader = RubyNeuralNets::DataLoaders::NumoImageMagick.new(
          dataset: 'test_simple_dataset',
          max_minibatch_size: 1,
          dataset_seed: 42,
          nbr_clones: 1,
          rot_angle: 0.0,
          grayscale: true,
          adaptive_invert: false,
          trim: false,
          resize: [28, 28],
          noise_intensity: 0.0,
          minmax_normalize: false
        )

        # For each dataset type, validate the minibatches
        [:training, :dev, :test].each do |dataset_type|
          dataset = data_loader.dataset(dataset_type)

          # Iterate through all minibatches
          dataset.each do |minibatch|
            # minibatch is Minibatches::Numo
            # x shape [784, 1], y shape [3, 1]

            # Since max_minibatch_size=1, only 1 sample
            x = minibatch.x[true, 0]  # shape [784]
            y = minibatch.y[true, 0]  # shape [3]

            # Find the class from one-hot y
            predicted_class = y.max_index

            # Check that the pixel values match the expected for the class
            expected_pixel = case predicted_class
                             when 0
                               0.0 / 65535.0
                             when 1
                               100.0 / 65535.0
                             when 2
                               200.0 / 65535.0
                             end

            # All pixels should be the expected value (approximately, due to normalization)
            expect((x - expected_pixel).abs.max < 0.01).to be true
          end
        end
      end
    end
  end
end
