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
      max_epochs.times do |idx_epoch|
        experiments.each do |experiment|
          if idx_epoch < experiment.nbr_epochs
            experiment.profiler.profile(idx_epoch) do
              log_prefix = "[Epoch #{idx_epoch}] [Exp #{experiment.exp_id}]"
              log "#{log_prefix} Start epoch training..."
              experiment.optimizer.start_epoch(idx_epoch)
              experiment.dataset.prepare_for_epoch
              idx_minibatch = 0
              experiment.dataset.each do |minibatch_x, (minibatch_y, minibatch_size)|
                minibatch_log_prefix = "#{log_prefix} [Minibatch #{idx_minibatch}]"
                log "#{minibatch_log_prefix} Retrieved minibatch of size #{minibatch_size}"
                debug { "#{minibatch_log_prefix} Minibatch X input: #{data_to_str(minibatch_x)}" }
                debug { "#{minibatch_log_prefix} Minibatch Y reference: #{data_to_str(minibatch_y)}" }
                experiment.optimizer.start_minibatch(idx_minibatch) if experiment.training_mode

                # Forward propagation
                experiment.model.initialize_back_propagation_cache
                saved_parameters = {}
                debug do
                  "#{minibatch_log_prefix} Model parameters:\n#{experiment.model.parameters.map do |p|
                    saved_parameters[p.name] = p.values.dup
                    "* #{p.name}: #{data_to_str(p.values)}"
                  end.join("\n")}"
                end
                a = experiment.model.forward_propagate(minibatch_x, train: experiment.training_mode)
                back_propagation_cache = experiment.training_mode ? experiment.model.back_propagation_cache : nil
                # Make sure other processing like gradient checking won't modify the cache again
                experiment.model.initialize_back_propagation_cache

                # Compute the loss for the minibatch
                loss = experiment.loss.compute_loss(a, minibatch_y)
                debug { "#{minibatch_log_prefix} Loss computed: #{data_to_str(loss)}" }

                # Display progress
                @progress_tracker.progress(experiment, idx_epoch, idx_minibatch, minibatch_x, minibatch_y, a, loss, minibatch_size)

                # Back propagation and gradient descent
                if experiment.training_mode
                  experiment.gradient_checker.check_gradients_for(idx_epoch, minibatch_x, minibatch_y) do
                    # Make sure gradient descent uses caches computed by the normal forward propagation
                    experiment.model.back_propagation_cache = back_propagation_cache
                    experiment.model.gradient_descent(experiment.loss.compute_loss_gradient(a, minibatch_y), a, minibatch_y, loss, minibatch_size)
                  end
                  experiment.optimizer.step
                  debug do
                    <<~EO_Debug
                      #{minibatch_log_prefix} Model parameters gradients:
                      #{experiment.model.parameters.map { |p| "* #{p.name}: #{data_to_str(p.values - saved_parameters[p.name])}" }.join("\n")}
                    EO_Debug
                  end
                end
                idx_minibatch += 1
              end
            end
          end
        end
      end
    end

  end

end
