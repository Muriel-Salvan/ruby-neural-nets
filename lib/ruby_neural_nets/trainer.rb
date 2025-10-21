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
    # * *model* (Model): The model to be trained
    # * *data_loader* (DataLoader): The data loader providing data for training
    # * *display_units* (Hash<Symbol, String or Regexp, Integer>): For each parameter name (or regexp matching name), indicate the number of units we want to picture [default: {}]
    def train(model, data_loader, display_units: {})
      log "Train on #{@nbr_epochs} epochs"
      data_loader.select_dataset_type(:training) do
        @progress_tracker.track('main', model, data_loader.labels, @loss, @accuracy, display_units: display_units) do
          @gradient_checker.link_to_model(model, @loss)
          @optimizer.teach_parameters(model.parameters)
          @nbr_epochs.times do |idx_epoch|
            @profiler.profile(idx_epoch) do
              log "Training for epoch ##{idx_epoch}..."
              @optimizer.start_epoch(idx_epoch)
              idx_minibatch = 0
              data_loader.each_minibatch do |minibatch_x, minibatch_y, minibatch_size|
                log "Retrieved minibatch ##{idx_minibatch} of size #{minibatch_size}"
                debug { "Minibatch X input: #{data_to_str(minibatch_x)}" }
                debug { "Minibatch Y reference: #{data_to_str(minibatch_y)}" }
                @optimizer.start_minibatch(idx_minibatch)
                # Forward propagation
                model.initialize_back_propagation_cache
                debug { "Model parameters:\n#{model.parameters.map { |p| "* #{p.name}: #{data_to_str(p.values)}" }.join("\n")}" }
                a = model.forward_propagate(minibatch_x, train: true)
                back_propagation_cache = model.back_propagation_cache
                # Make sure other processing like gradient checking won't modify the cache again
                model.initialize_back_propagation_cache
                # Compute the loss for the minibatch
                loss = @loss.compute_loss(a, minibatch_y)
                debug { "Loss computed: #{data_to_str(loss)}" }
                # Display progress
                @progress_tracker.progress('main', idx_epoch, idx_minibatch, minibatch_x, minibatch_y, a, loss, minibatch_size)
                # Gradient descent
                @gradient_checker.check_gradients_for(idx_epoch, minibatch_x, minibatch_y) do
                  # Make sure gradient descent uses caches computed by the normal forward propagation
                  model.back_propagation_cache = back_propagation_cache
                  model.gradient_descent(@loss.compute_loss_gradient(a, minibatch_y), a, minibatch_y, loss, minibatch_size)
                end
                @optimizer.step
                idx_minibatch += 1
              end
            end
          end
        end
      end
    end

  end

end
