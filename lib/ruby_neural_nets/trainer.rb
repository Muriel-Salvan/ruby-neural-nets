require 'ruby_neural_nets/accuracy'
require 'ruby_neural_nets/gradient_checker'
require 'ruby_neural_nets/logger'
require 'ruby_neural_nets/losses/cross_entropy'
require 'ruby_neural_nets/profiler'
require 'ruby_neural_nets/progress_tracker'
require 'ruby_neural_nets/datasets/minibatch'
require 'ruby_neural_nets/helpers'

module RubyNeuralNets

  class Trainer
    include Logger

    # Constructor
    #
    # Parameters::
    # * *progress_tracker* (ProgressTracker): The gradients checker to be used [default: ProgressTracker.new]
    def initialize(progress_tracker: ProgressTracker.new)
      @progress_tracker = progress_tracker
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

      @progress_tracker.session do
        # Track dev losses per epoch
        dev_losses = {}
        # Track early stopping state per training experiment
        early_stopping_state = {}
        max_epochs.times do |idx_epoch|
          # Evaluate on dev experiments first
          dev_experiments.each do |dev_exp|
            result = process_experiment(dev_exp, idx_epoch, false)
            dev_losses[dev_exp.exp_id] = result[:loss] if result
          end

          # Train on training experiments
          training_experiments.each do |training_exp|
            process_experiment(training_exp, idx_epoch, true)

            # Check early stopping for this training experiment
            if training_exp.dev_experiment
              dev_loss = dev_losses[training_exp.dev_experiment.exp_id]
              if dev_loss
                state = early_stopping_state[training_exp.exp_id] ||= { best_loss: Float::INFINITY, epochs_without_improvement: 0, early_stopping_reached: false }
                unless state[:early_stopping_reached]
                  if dev_loss < state[:best_loss]
                    state[:best_loss] = dev_loss
                    state[:epochs_without_improvement] = 0
                    debug { "[Epoch #{idx_epoch}] [Exp #{training_exp.exp_id}] New best dev loss: #{dev_loss}" }
                  else
                    state[:epochs_without_improvement] += 1
                    debug { "[Epoch #{idx_epoch}] [Exp #{training_exp.exp_id}] No improvement in dev loss for #{state[:epochs_without_improvement]} epochs" }
                  end

                  if state[:epochs_without_improvement] >= training_exp.early_stopping_patience
                    debug { "[Epoch #{idx_epoch}] [Exp #{training_exp.exp_id}] Early stopping reached at epoch #{idx_epoch}" }
                    @progress_tracker.notify_early_stopping(training_exp, idx_epoch)
                    @progress_tracker.notify_early_stopping(training_exp.dev_experiment, idx_epoch)
                    # Stop tracking early stopping for this experiment
                    state[:early_stopping_reached] = true
                  end
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
    # * Hash or nil: Hash with :loss and :accuracy keys, or nil if not applicable
    def process_experiment(experiment, idx_epoch, is_training)
      result = nil
      if idx_epoch < experiment.nbr_epochs
        experiment.profiler.profile(idx_epoch) do
          log_prefix = "[Epoch #{idx_epoch}] [Exp #{experiment.exp_id}]"
          log "#{log_prefix} Start epoch #{is_training ? 'training' : 'evaluation'} on #{experiment.model.parameters.map(&:size).sum} parameters..."
          experiment.optimizer.start_epoch(idx_epoch) if is_training
          experiment.dataset.prepare_for_epoch
          total_loss = 0.0
          total_accuracy = 0.0
          total_size = 0
          experiment.dataset.each.with_index do |minibatch, idx_minibatch|
            minibatch_log_prefix = "#{log_prefix} [Minibatch #{idx_minibatch}]"
            log "#{minibatch_log_prefix} Retrieved minibatch of size #{minibatch.size}"
            debug { "#{minibatch_log_prefix} Minibatch X input: #{data_to_str(minibatch.x)}" }
            debug { "#{minibatch_log_prefix} Minibatch Y reference: #{data_to_str(minibatch.y)}" }
            # Add code to dump minibatches if the experiment option is enabled
            if experiment.dump_minibatches
              # For each element in the minibatch
              # Assuming the minibatch has an each_element method
              minibatch.each_element.with_index do |(element, y), idx_element|
                # Write the image using Helpers
                Helpers.write_image(
                  experiment.dataset.to_image(element),
                  "./minibatches/#{experiment.exp_id}/#{idx_epoch}/#{idx_minibatch}/#{idx_element}_#{experiment.dataset.underlying_label(y)}.png"
                )
              end
            end
            experiment.optimizer.start_minibatch(idx_minibatch, minibatch.size) if is_training

            # Forward propagation
            experiment.model.initialize_back_propagation_cache
            saved_parameters = {}
            debug do
              "#{minibatch_log_prefix} Model parameters:\n#{experiment.model.parameters.map do |p|
                saved_parameters[p.name] = p.values.dup
                "* #{p.name}: #{data_to_str(p.values)}"
              end.join("\n")}"
            end
            a = experiment.model.forward_propagate(minibatch.x, train: is_training)
            back_propagation_cache = is_training ? experiment.model.back_propagation_cache : nil
            # Make sure other processing like gradient checking won't modify the cache again
            experiment.model.initialize_back_propagation_cache

            # Compute the loss for the minibatch (including L2 regularization if applicable)
            loss = experiment.loss.compute_loss(a, minibatch.y, experiment.model)
            debug { "#{minibatch_log_prefix} Loss computed: #{data_to_str(loss)}" }

            # Display progress
            @progress_tracker.progress(experiment, idx_epoch, idx_minibatch, minibatch, a, loss)

            # Back propagation and gradient descent if training
            if is_training
              experiment.gradient_checker.check_gradients_for(idx_epoch, minibatch) do
                # Make sure gradient descent uses caches computed by the normal forward propagation
                experiment.model.back_propagation_cache = back_propagation_cache
                experiment.model.gradient_descent(experiment.loss.compute_loss_gradient(a, minibatch.y, experiment.model), a, minibatch, loss)
              end
              experiment.optimizer.step
              debug do
                <<~EO_Debug
                  #{minibatch_log_prefix} Model parameters gradients:
                  #{experiment.model.parameters.map { |p| "* #{p.name}: #{data_to_str(p.values - saved_parameters[p.name])}" }.join("\n")}"
                EO_Debug
              end
            end

            # Accumulate loss and accuracy for evaluation
            total_loss += loss.mean * minibatch.size
            accuracy = experiment.accuracy.measure(a, minibatch)
            total_accuracy += accuracy * minibatch.size
            total_size += minibatch.size
          end
          average_loss = total_loss / total_size
          average_accuracy = total_accuracy / total_size
          result = { loss: average_loss, accuracy: average_accuracy }
        end
      end
      result
    end

  end

end
