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

  let(:progress_tracker) { RubyNeuralNets::ProgressTracker.new(display_graphs: false) }
  let(:trainer) { RubyNeuralNets::Trainer.new(progress_tracker: progress_tracker) }

  let(:model) { RubyNeuralNets::Models::OneLayer.new(28, 28, 1, 3) }
  let(:loss) { RubyNeuralNets::Losses::CrossEntropy.new }
  let(:optimizer) { RubyNeuralNets::Optimizers::Adam.new(learning_rate: 0.01, weight_decay: 0.0) }
  let(:accuracy) { RubyNeuralNets::Accuracies::ClassesNumo.new }

  let(:profiler) { RubyNeuralNets::Profiler.new(profiling: false) }
  let(:gradient_checker) { RubyNeuralNets::GradientChecker.new(gradient_checks: :off) }

  describe '#train' do
    it 'tracks progress and reports correct cost and accuracy' do
      # Define the mocked filesystem
      files = {}
      (0..2).each do |class_idx|
        (0..9).each do |img_idx|
          # Store the PNG data
          files["test_rspec_dataset/#{class_idx}/test_image_#{img_idx + class_idx * 10}.png"] = png(
            28,
            28,
            { color: [
              case class_idx
              when 0
                0
              when 1
                100
              when 2
                200
              else
                50
              end
            ] }
          )
        end
      end

      with_test_dir(files) do |datasets_path|
        # Create data loader inside fakefs
        data_loader = RubyNeuralNets::DataLoaders::NumoImageMagick.new(
          datasets_path: datasets_path,
          dataset: 'test_rspec_dataset',
          max_minibatch_size: 2,
          dataset_seed: 42,
          partitions: { training: 0.7, dev: 0.15, test: 0.15 },
          nbr_clones: 1,
          rot_angle: 0.0,
          grayscale: true,
          adaptive_invert: false,
          trim: false,
          resize: [28, 28],
          noise_intensity: 0.0,
          minmax_normalize: true,
          video_slices_sec: 1.0,
          filter_dataset: 'all'
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

    it 'captures log output in test helper' do
      # Test that logging is captured and accessible via helper
      # Create a simple experiment mock
      experiment = double('experiment', exp_id: 'test_exp')
      progress_tracker.instance_variable_set(:@experiments, { 'test_exp' => { experiment: experiment } })
      progress_tracker.notify_early_stopping(experiment, 1)

      log_output = captured_log_output
      expect(log_output).to include('Early stopping notified for [Exp test_exp] at epoch 1')
    end
  end
end