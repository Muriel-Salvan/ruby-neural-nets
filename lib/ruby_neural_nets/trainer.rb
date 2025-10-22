require 'ruby_neural_nets/accuracy'
require 'ruby_neural_nets/gradient_checker'
require 'ruby_neural_nets/logger'
require 'ruby_neural_nets/losses/cross_entropy'
require 'ruby_neural_nets/profiler'
require 'ruby_neural_nets/progress_tracker'
require 'ruby_neural_nets/datasets/minibatch'

module RubyNeuralNets

  class Trainer
    include Logger

    # Constructor
    #
    # Parameters::
    # * *nbr_epochs* (Integer): Number of epochs to use for training
    # * *accuracy* (Accuracy): The accuracy measure [default: Accuracy.new]
    # * *loss* (Loss): The loss measure [default: Losses::CrossEntropy.new]
    # * *optimizer* (Optimizer): The optimizer to be used [default: Optimizers::Constant.new(learning_rate: 0.001)]
    # * *gradient_checker* (GradientChecker): The gradients checker to be used [default: GradientChecker.new]
    # * *progress_tracker* (ProgressTracker): The gradients checker to be used [default: ProgressTracker.new]
    # * *profiler* (Profiler): The profiler to be used [default: Profiler.new]
    def initialize(
      nbr_epochs:,
      accuracy: Accuracy.new,
      loss: Losses::CrossEntropy.new,
      optimizer: Optimizers::Constant.new(learning_rate: 0.001),
      gradient_checker: GradientChecker.new,
      progress_tracker: ProgressTracker.new,
      profiler: Profiler.new
    )
      @nbr_epochs = nbr_epochs
      @accuracy = accuracy
      @loss = loss
      @optimizer = optimizer
      @gradient_checker = gradient_checker
      @progress_tracker = progress_tracker
      @profiler = profiler
    end

    # Train a given model on a training dataset
    #
    # Parameters::
    # * *experiment_id* (String): The experiment ID to track progress for
    # * *model* (Model): The model to be trained
    # * *dataset_training* (Dataset): The dataset providing data for training
    # * *dataset_dev* (Dataset or nil): The dataset providing data for dev, or nil if only used for training [default: nil]
    def train(experiment_id, model, dataset_training, dataset_dev: nil)
      log "Train on #{@nbr_epochs} epochs"
      @gradient_checker.link_to_model(model, @loss)
      @optimizer.teach_parameters(model.parameters)
      @nbr_epochs.times do |idx_epoch|
        @profiler.profile(idx_epoch) do
          log "Training for epoch ##{idx_epoch}..."
          @optimizer.start_epoch(idx_epoch)
          process_dataset(experiment_id, idx_epoch, model, dataset_training, true)
          process_dataset("#{experiment_id}_dev", idx_epoch, model, dataset_dev, false) unless dataset_dev.nil?
        end
      end
    end

    private

    # Process a dataset.
    #
    # Parameters::
    # * *experiment_id* (String): Experiment ID to be used for tracking
    # * *idx_epoch* (Integer): The epoch index
    # * *model* (Model): Model to be used for processing
    # * *dataset* (Dataset): Dataset to be processed
    # * *train* (Boolean): Are we in training mode?
    def process_dataset(experiment_id, idx_epoch, model, dataset, train)
      dataset.prepare_for_epoch
      selected_partition = dataset.selected_partition
      idx_minibatch = 0
      dataset.each do |minibatch_x, (minibatch_y, minibatch_size)|
        log "[#{selected_partition}] Retrieved minibatch ##{idx_minibatch} of size #{minibatch_size}"
        debug { "[#{selected_partition}] Minibatch X input: #{data_to_str(minibatch_x)}" }
        debug { "[#{selected_partition}] Minibatch Y reference: #{data_to_str(minibatch_y)}" }
        @optimizer.start_minibatch(idx_minibatch) if train
        # Forward propagation
        model.initialize_back_propagation_cache
        debug { "[#{selected_partition}] Model parameters:\n#{model.parameters.map { |p| "* #{p.name}: #{data_to_str(p.values)}" }.join("\n")}" }
        a = model.forward_propagate(minibatch_x, train:)
        back_propagation_cache = train ? model.back_propagation_cache : nil
        # Make sure other processing like gradient checking won't modify the cache again
        model.initialize_back_propagation_cache
        # Compute the loss for the minibatch
        loss = @loss.compute_loss(a, minibatch_y)
        debug { "[#{selected_partition}] Loss computed: #{data_to_str(loss)}" }
        # Display progress
        @progress_tracker.progress(experiment_id, idx_epoch, idx_minibatch, minibatch_x, minibatch_y, a, loss, minibatch_size)
        if train
          # Gradient descent
          @gradient_checker.check_gradients_for(idx_epoch, minibatch_x, minibatch_y) do
            # Make sure gradient descent uses caches computed by the normal forward propagation
            model.back_propagation_cache = back_propagation_cache
            model.gradient_descent(@loss.compute_loss_gradient(a, minibatch_y), a, minibatch_y, loss, minibatch_size)
          end
          @optimizer.step
        end
        idx_minibatch += 1
      end
    end

  end

end
