require 'rspec'
require 'tempfile'
require 'stringio'
require 'numo/narray'
require 'rmagick'

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

describe RubyNeuralNets::Trainer do
  # Store original methods in class variables
  before(:all) do
    # @@original_dir_glob = Dir.method(:glob)
    # @@original_image_list_new = Magick::ImageList.method(:new)
    # @@original_file_directory = File.method(:directory?)

    # Mock Dir.glob to return test files for our test dataset
    Dir.define_singleton_method(:glob) do |pattern|
      if pattern.include?('./datasets/test_rspec_dataset/') || pattern.include?('./datasets\\test_rspec_dataset\\')
        # Return mock directory structure for labels
        if pattern == './datasets/test_rspec_dataset/*'
          ['./datasets/test_rspec_dataset/0', './datasets/test_rspec_dataset/1', './datasets/test_rspec_dataset/2']
        elsif pattern == './datasets/test_rspec_dataset/0/*'
          (0..9).map { |i| "./datasets/test_rspec_dataset/0/test_image_#{i}.png" }
        elsif pattern == './datasets/test_rspec_dataset/1/*'
          (10..19).map { |i| "./datasets/test_rspec_dataset/1/test_image_#{i}.png" }
        elsif pattern == './datasets/test_rspec_dataset/2/*'
          (20..29).map { |i| "./datasets/test_rspec_dataset/2/test_image_#{i}.png" }
        else
          []
        end
      else
        # For other patterns, call the original method - just use Dir.glob directly
        Dir.glob(pattern)
      end
    end

    # Mock File.directory? to return true for our test directories
    File.define_singleton_method(:directory?) do |path|
      if path.start_with?('./datasets/test_rspec_dataset/')
        true
      else
        File.directory?(path)
      end
    end

    # Mock Magick::ImageList.new to return test data
    Magick::ImageList.define_singleton_method(:new) do |*args|
      if args.first && (args.first.include?('/test_rspec_dataset/') || args.first.include?("\\test_rspec_dataset\\"))
        # Create a real ImageList but populate it with mock images containing test data
        image_list = Magick::ImageList.allocate
        image_list.send(:initialize)

        # Create a real Magick::Image with test data
        mock_image = Magick::Image.new(28, 28)

        # Generate deterministic pixel data based on the class (directory)
        pixel_data = []
        if args.first.include?('/0/') || args.first.include?('\\0\\')
          # Class 0: simple pattern
          pixel_data = Array.new(28 * 28) { |j| (j % 256) }
        elsif args.first.include?('/1/') || args.first.include?('\\1\\')
          # Class 1: different pattern
          pixel_data = Array.new(28 * 28) { |j| ((j + 100) % 256) }
        elsif args.first.include?('/2/') || args.first.include?('\\2\\')
          # Class 2: another pattern
          pixel_data = Array.new(28 * 28) { |j| ((j + 200) % 256) }
        else
          # Default pattern
          pixel_data = Array.new(28 * 28) { |j| ((j + 50) % 256) }
        end

        # Import the pixel data into the image
        mock_image.import_pixels(0, 0, 28, 28, 'I', pixel_data)
        image_list << mock_image

        image_list
      else
        # For other cases, create a new ImageList without calling the mocked method
        # Use allocate to bypass the constructor
        image_list = Magick::ImageList.allocate
        image_list.send(:initialize, *args)
        image_list
      end
    end
  end

  # after(:all) do
  #   # Restore original methods
  #   Dir.define_singleton_method(:glob, @@original_dir_glob)
  #   File.define_singleton_method(:directory?, @@original_file_directory)
  #   Magick::ImageList.define_singleton_method(:new, @@original_image_list_new)
  # end
  
  let(:progress_tracker) { RubyNeuralNets::ProgressTracker.new(display_graphs: false) }
  let(:trainer) { RubyNeuralNets::Trainer.new(progress_tracker: progress_tracker) }
  
  let(:data_loader) { @data_loader }
  
  let(:model) { RubyNeuralNets::Models::OneLayer.new(28, 28, 1, 3) }
  let(:loss) { RubyNeuralNets::Losses::CrossEntropy.new }
  let(:optimizer) { RubyNeuralNets::Optimizers::Adam.new(learning_rate: 0.01, weight_decay: 0.0) }
  let(:accuracy) { RubyNeuralNets::Accuracies::ClassesNumo.new }
  
  before do
    @data_loader = RubyNeuralNets::DataLoaders::NumoImageMagick.new(
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
  end

  let(:profiler) { RubyNeuralNets::Profiler.new(profiling: false) }
  let(:gradient_checker) { RubyNeuralNets::GradientChecker.new(gradient_checks: :off) }

  let(:experiment) do
    RubyNeuralNets::Experiment.new(
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
  end
  
  describe '#train' do
    it 'tracks progress and reports correct cost and accuracy using real data loader with mocked file access' do
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
