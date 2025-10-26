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
    # * *progress_tracker* (ProgressTracker): The gradients checker to be used [default: ProgressTracker.new]
    # * *early_stopping_patience* (Integer): Number of epochs to wait for dev accuracy improvement before stopping [default: 10]
    def initialize(progress_tracker: ProgressTracker.new, early_stopping_patience: 10)
      @progress_tracker = progress_tracker
      @early_stopping_patience = early_stopping_patience
    end

    # Train experiments
    #
    # Parameters::
    # * *experiments* (Array<Experiment>): The experiments to be trained (or evaluated)
    def train(experiments)
      max_epochs = experiments.map(&:nbr_epochs).max
      log "Train #{experiments.size} experiments on #{max_epochs} epochs maximum"

      # Identify training and dev experiments
      training_experiments = experiments.select(&:training_mode)
      dev_experiments = experiments.reject(&:training_mode)

      # Track dev accuracies per epoch
      dev_accuracies = {}

      # Track early stopping state per training experiment
      early_stopping_state = {}

      max_epochs.times do |idx_epoch|
        # Evaluate on dev experiments first
        dev_experiments.each do |dev_exp|
          dev_accuracies[dev_exp.exp_id] = process_experiment(dev_exp, idx_epoch, false)
        end

        # Train on training experiments
        training_experiments.each do |training_exp|
          process_experiment(training_exp, idx_epoch, true)

          # Check early stopping for this training experiment
          if training_exp.dev_experiment
            dev_accuracy = dev_accuracies[training_exp.dev_experiment.exp_id]
            if dev_accuracy
              state = early_stopping_state[training_exp.exp_id] ||= { best_accuracy: 0.0, epochs_without_improvement: 0, early_stopping_reached: false }
              unless state[:early_stopping_reached]
                if dev_accuracy > state[:best_accuracy]
                  state[:best_accuracy] = dev_accuracy
                  state[:epochs_without_improvement] = 0
                  log "[Epoch #{idx_epoch}] [Exp #{training_exp.exp_id}] New best dev accuracy: #{dev_accuracy * 100}%"
                else
                  state[:epochs_without_improvement] += 1
                  log "[Epoch #{idx_epoch}] [Exp #{training_exp.exp_id}] No improvement in dev accuracy for #{state[:epochs_without_improvement]} epochs"
                end

                if state[:epochs_without_improvement] >= @early_stopping_patience
                  log "[Epoch #{idx_epoch}] [Exp #{training_exp.exp_id}] Early stopping reached at epoch #{idx_epoch}"
                  @progress_tracker.notify_early_stopping(training_exp, idx_epoch)
                  # Stop tracking early stopping for this experiment
                  state[:early_stopping_reached] = true
                end
              end
            end
          end
        end
      end
    end

    private

    # Process a single experiment for an epoch
    #
    # Parameters::
    # * *experiment* (Experiment): The experiment to process
    # * *idx_epoch* (Integer): Current epoch index
    # * *is_training* (Boolean): Whether to perform training or evaluation
    # Result::
    # * Float or nil: The average accuracy for the epoch (if applicable)
    def process_experiment(experiment, idx_epoch, is_training)
      average_accuracy = nil
      if idx_epoch < experiment.nbr_epochs
        experiment.profiler.profile(idx_epoch) do
          log_prefix = "[Epoch #{idx_epoch}] [Exp #{experiment.exp_id}]"
          log "#{log_prefix} Start epoch #{is_training ? 'training' : 'evaluation'}..."
          experiment.optimizer.start_epoch(idx_epoch) if is_training
          experiment.dataset.prepare_for_epoch
          idx_minibatch = 0
          total_accuracy = 0.0
          total_size = 0
          experiment.dataset.each do |minibatch_x, (minibatch_y, minibatch_size)|
            minibatch_log_prefix = "#{log_prefix} [Minibatch #{idx_minibatch}]"
            log "#{minibatch_log_prefix} Retrieved minibatch of size #{minibatch_size}"
            debug { "#{minibatch_log_prefix} Minibatch X input: #{data_to_str(minibatch_x)}" }
            debug { "#{minibatch_log_prefix} Minibatch Y reference: #{data_to_str(minibatch_y)}" }
            experiment.optimizer.start_minibatch(idx_minibatch, minibatch_size) if is_training

            # Forward propagation
            experiment.model.initialize_back_propagation_cache
            saved_parameters = {}
            debug do
              "#{minibatch_log_prefix} Model parameters:\n#{experiment.model.parameters.map do |p|
                saved_parameters[p.name] = p.values.dup
                "* #{p.name}: #{data_to_str(p.values)}"
              end.join("\n")}"
            end
            a = experiment.model.forward_propagate(minibatch_x, train: is_training)
            back_propagation_cache = is_training ? experiment.model.back_propagation_cache : nil
            # Make sure other processing like gradient checking won't modify the cache again
            experiment.model.initialize_back_propagation_cache

            # Compute the loss for the minibatch
            loss = experiment.loss.compute_loss(a, minibatch_y, experiment.model)
            debug { "#{minibatch_log_prefix} Loss computed: #{data_to_str(loss)}" }

            # Display progress
            @progress_tracker.progress(experiment, idx_epoch, idx_minibatch, minibatch_x, minibatch_y, a, loss, minibatch_size)

            # Back propagation and gradient descent if training
            if is_training
              experiment.gradient_checker.check_gradients_for(idx_epoch, minibatch_x, minibatch_y) do
                # Make sure gradient descent uses caches computed by the normal forward propagation
                experiment.model.back_propagation_cache = back_propagation_cache
                experiment.model.gradient_descent(experiment.loss.compute_loss_gradient(a, minibatch_y, experiment.model), a, minibatch_y, loss, minibatch_size)
              end
              experiment.optimizer.step
              debug do
                <<~EO_Debug
                  #{minibatch_log_prefix} Model parameters gradients:
                  #{experiment.model.parameters.map { |p| "* #{p.name}: #{data_to_str(p.values - saved_parameters[p.name])}" }.join("\n")}"
                EO_Debug
              end
            end
            # Accumulate accuracy for evaluation
            accuracy = experiment.accuracy.measure(a, minibatch_y, minibatch_size)
            total_accuracy += accuracy * minibatch_size
            total_size += minibatch_size
            idx_minibatch += 1
          end
          average_accuracy = total_accuracy / total_size
        end
      end
      average_accuracy
    end

  end

end
