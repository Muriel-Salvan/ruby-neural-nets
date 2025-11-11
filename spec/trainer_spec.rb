require 'rspec'
require 'tempfile'
require 'stringio'
require 'numo/narray'
require 'rmagick'
require 'fakefs/spec_helpers'
require 'fileutils'

# Add the lib directory to the load path
$LOAD_PATH.unshift(File.join(File.dirname(__FILE__), '..', 'lib'))

# Require the necessary modules
require 'ruby_neural_nets/trainer'
require 'ruby_neural_nets/models/one_layer'
require 'ruby_neural_nets/data_loaders/numo_image_magick'
require 'ruby_neural_nets/progress_tracker'
require 'ruby_neural_nets/experiment'
require 'ruby_neural_nets/losses/cross_entropy'
require 'ruby_neural_nets/optimizers/adam'
require 'ruby_neural_nets/accuracies/classes_numo'
require 'ruby_neural_nets/profiler'
require 'ruby_neural_nets/gradient_checker'

# Helper method to setup mocked filesystem using fakefs
def setup_test_filesystem(files_hash)
  files_hash.each do |path, content|
    dir = File.dirname(path)
    FileUtils.mkdir_p(dir) unless Dir.exist?(dir)
    File.write(path, content)
  end
end

describe RubyNeuralNets::Trainer do

  let(:progress_tracker) { RubyNeuralNets::ProgressTracker.new(display_graphs: false) }
  let(:trainer) { RubyNeuralNets::Trainer.new(progress_tracker: progress_tracker) }

  let(:model) { RubyNeuralNets::Models::OneLayer.new(28, 28, 1, 3) }
  let(:loss) { RubyNeuralNets::Losses::CrossEntropy.new }
  let(:optimizer) { RubyNeuralNets::Optimizers::Adam.new(learning_rate: 0.01, weight_decay: 0.0) }
  let(:accuracy) { RubyNeuralNets::Accuracies::ClassesNumo.new }

  let(:profiler) { RubyNeuralNets::Profiler.new(profiling: false) }
  let(:gradient_checker) { RubyNeuralNets::GradientChecker.new(gradient_checks: :off) }
  
  describe '#train' do
    it 'tracks progress and reports correct cost and accuracy using real data loader with mocked file access' do
      FakeFS do
        # Define the mocked filesystem
        files = {}
        (0..2).each do |class_idx|
          (0..9).each do |img_idx|
            file_path = "datasets/test_rspec_dataset/#{class_idx}/test_image_#{img_idx + class_idx * 10}.png"

            # Create a real Magick::Image with test data
            mock_image = Magick::Image.new(28, 28) do |img|
              img.format = 'PNG'
            end

            # Generate deterministic pixel data based on the class
            pixel_data = Array.new(28 * 28) do |j|
              case class_idx
              when 0
                (j % 256)
              when 1
                ((j + 100) % 256)
              when 2
                ((j + 200) % 256)
              else
                ((j + 50) % 256)
              end
            end

            # Import the pixel data into the image
            mock_image.import_pixels(0, 0, 28, 28, 'I', pixel_data)

            # Store the PNG data
            files[file_path] = mock_image.to_blob
          end
        end

        setup_test_filesystem(files)

        # Mock Magick::ImageList.new to return images from the fakefs files
        Magick::ImageList.define_singleton_method(:new) do |*args|
          if args.first && (args.first.include?('test_rspec_dataset'))
            # Read the PNG data from fakefs
            png_data = File.read(args.first)
            # Create image from blob
            image = Magick::Image.from_blob(png_data).first
            # Create ImageList and add the image
            image_list = Magick::ImageList.allocate
            image_list.send(:initialize)
            image_list << image
            image_list
          else
            # For other cases, create a new ImageList normally
            Magick::ImageList.allocate.send(:initialize, *args)
          end
        end

        # Create data loader inside fakefs
        data_loader = RubyNeuralNets::DataLoaders::NumoImageMagick.new(
          dataset: 'test_rspec_dataset',
          max_minibatch_size: 2,
          dataset_seed: 42,
          nbr_clones: 1,
          rot_angle: 0.0,
          grayscale: true,
          adaptive_invert: false,
          trim: false,
          resize: [28, 28],
          noise_intensity: 0.0,
          minmax_normalize: true
        )

        # Create experiment inside fakefs
        experiment = RubyNeuralNets::Experiment.new(
          exp_id: 'test_rspec_experiment',
          dataset: data_loader.dataset(:training),
          model: model,
          data_loader: data_loader,
          loss: loss,
          optimizer: optimizer,
          accuracy: accuracy,
          nbr_epochs: 2,
          training_mode: true,
          profiler: profiler,
          gradient_checker: gradient_checker
        )

        # Link optimizer to model parameters
        optimizer.teach_parameters(model.parameters)

        # Track the experiment
        progress_tracker.track(experiment)

        # Train the experiment
        trainer.train([experiment])

        # Get the epochs data from the progress tracker
        epochs_data = progress_tracker.epochs_data(experiment)

        # Verify that epochs data was recorded
        expect(epochs_data).not_to be_empty
        expect(epochs_data.keys).to include(0, 1) # Both epochs should be present

        # Verify that each epoch has minibatch data
        epochs_data.each do |epoch_idx, epoch_data|
          expect(epoch_data).not_to be_empty

          # Verify each minibatch has cost and accuracy
          epoch_data.each do |minibatch_idx, minibatch_data|
            expect(minibatch_data).to have_key(:cost)
            expect(minibatch_data).to have_key(:accuracy)
            expect(minibatch_data[:cost]).to be_a(Numeric)
            expect(minibatch_data[:accuracy]).to be_a(Numeric)

            # Verify accuracy is between 0 and 1
            expect(minibatch_data[:accuracy]).to be_between(0, 1)
          end
        end
      end
    end
  end
end
