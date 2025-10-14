require 'ruby_neural_nets/accuracy'
require 'ruby_neural_nets/gradient_checker'
require 'ruby_neural_nets/losses/cross_entropy'
require 'ruby_neural_nets/profiler'
require 'ruby_neural_nets/progress_tracker'

module RubyNeuralNets

  class Trainer

    # Constructor
    #
    # Parameters::
    # * *nbr_epochs* (Integer): Number of epochs to use for training
    # * *max_minibatch_size* (Integer): Max size each minibatch should have
    # * *accuracy* (Accuracy): The accuracy measure [default: Accuracy.new]
    # * *loss* (Loss): The loss measure [default: Losses::CrossEntropy.new]
    # * *optimizer* (Optimizer): The optimizer to be used [default: Optimizers::Constant.new(learning_rate: 0.001)]
    # * *gradient_checker* (GradientChecker): The gradients checker to be used [default: GradientChecker.new]
    # * *progress_tracker* (ProgressTracker): The gradients checker to be used [default: ProgressTracker.new]
    # * *profiler* (Profiler): The profiler to be used [default: Profiler.new]
    def initialize(
      nbr_epochs:,
      max_minibatch_size:,
      accuracy: Accuracy.new,
      loss: Losses::CrossEntropy.new,
      optimizer: Optimizers::Constant.new(learning_rate: 0.001),
      gradient_checker: GradientChecker.new,
      progress_tracker: ProgressTracker.new,
      profiler: Profiler.new
    )
      @nbr_epochs = nbr_epochs
      @max_minibatch_size = max_minibatch_size
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
    # * *dataset* (Dataset): The dataset on which the model has to be trained
    # * *dataset_type* (Symbol): The dataset type on which the model has to be trained [default: :train]
    def train(model, dataset, dataset_type: :train)
      puts "[Trainer] - Train with minibatches of size #{@max_minibatch_size}, on #{@nbr_epochs} epochs"
      @progress_tracker.track(model, dataset.classes, @loss, @accuracy) do
        @gradient_checker.link_to_model(model, @loss)
        @optimizer.teach_parameters(model.parameters)
        @nbr_epochs.times do |idx_epoch|
          @profiler.profile(idx_epoch) do
            puts "[Trainer] - Training for epoch ##{idx_epoch}..."
            @optimizer.start_epoch(idx_epoch)
            idx_minibatch = 0
            dataset.for_each_minibatch(dataset_type, @max_minibatch_size) do |minibatch_x, minibatch_y, minibatch_size|
              @optimizer.start_minibatch(idx_minibatch)
              # Forward propagation
              model.initialize_back_propagation_cache
              a = model.forward_propagate(minibatch_x, train: true)
              back_propagation_cache = model.back_propagation_cache
              # Make sure other processing like gradient checking won't modify the cache again
              model.initialize_back_propagation_cache
              # Compute the loss for the minibatch
              loss = @loss.compute_loss(a, minibatch_y)
              # Display progress
              @progress_tracker.progress(idx_epoch, idx_minibatch, minibatch_x, minibatch_y, a, loss, minibatch_size)
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
