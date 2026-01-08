require 'ruby_neural_nets/logger'
require 'ruby_neural_nets/profiler'
require 'ruby_neural_nets/helpers'

module RubyNeuralNets

  # Class dedicated to infering models on datasets.
  # This class handles forward propagation without any concept of epochs, loss, accuracy or gradient checking.
  class Inferer
    include Logger

    # Run inference on an experiment
    #
    # Parameters::
    # * *experiment* (Experiment): The experiment to run inference on
    # * *idx_epoch* (Integer): Current epoch index
    # * *block* (Proc): Block that receives inferred data for each minibatch
    #   * Parameters::
    #     * *minibatch* (Minibatch): The current minibatch being processed
    #     * *a* (Object): The model's output (predictions)
    #     * *idx_minibatch* (Integer): The current minibatch index
    def infer(experiment, idx_epoch)
      log "Start inference on #{experiment.model.parameters.map(&:size).sum} parameters..."
      
      experiment.profiler.profile(idx_epoch) do
        log_prefix = "[Inference] [Exp #{experiment.exp_id}]"
        experiment.dataset.prepare_for_epoch
        
        experiment.dataset.each.with_index do |minibatch, idx_minibatch|
          minibatch_log_prefix = "#{log_prefix} [Minibatch #{idx_minibatch}]"
          log "#{minibatch_log_prefix} Retrieved minibatch of size #{minibatch.size}"
          debug { "#{minibatch_log_prefix} Minibatch X input: #{data_to_str(minibatch.x)}" }
          debug { "#{minibatch_log_prefix} Minibatch Y reference: #{data_to_str(minibatch.y)}" }
          
          # Add code to dump minibatches if experiment option is enabled
          if experiment.dump_minibatches
            # For each element in the minibatch
            minibatch.each_element.with_index do |(element, y), idx_element|
              # Write the image using Helpers
              Helpers.write_image(
                experiment.dataset.to_image(element),
                "./minibatches/#{experiment.exp_id}/#{idx_epoch}/#{idx_minibatch}/#{idx_element}_#{experiment.dataset.underlying_label(y)}.png"
              )
            end
          end

          # Prepare for minibatch processing
          prepare_for_minibatch(minibatch, experiment, idx_minibatch)
          
          debug do
            "#{minibatch_log_prefix} Model parameters:\n#{experiment.model.parameters.map do |p|
              "* #{p.name}: #{data_to_str(p.values)}"
            end.join("\n")}"
          end
          a = experiment.model.forward_propagate(minibatch.x, train: experiment.training_mode)

          debug { "#{minibatch_log_prefix} Model output: #{data_to_str(a)}" }

          # Yield back to the caller with inferred data
          yield minibatch, a, idx_minibatch
        end
      end
      
      log "Inference completed."
    end

    private

    # Prepare for processing a minibatch.
    # This method can be overridden by subclasses to perform specific preparation steps.
    #
    # Parameters::
    # * *minibatch* (Minibatch): The minibatch to prepare for
    # * *experiment* (Experiment): The experiment being processed
    # * *idx_minibatch* (Integer): The current minibatch index
    def prepare_for_minibatch(minibatch, experiment, idx_minibatch)
      # Do nothing by default
    end

  end

end      
