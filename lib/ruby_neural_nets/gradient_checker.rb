require 'numo/narray'
require 'ruby_neural_nets/helpers'
require 'ruby_neural_nets/logger'

module RubyNeuralNets

  class GradientChecker
    include Logger

    # Constructor
    #
    # Parameters::
    # * *gradient_checks* (Symbol): Behavior when wrong gradient checking is detected [default: :warning]
    #   Possible values are the same as for Helpers.instability_checks
    # * *nbr_gradient_checks_samples* (Integer): Max number of parameters per model's parameter tensor to be used for gradient checking [default: 2]
    # * *gradient_checks_epochs_interval* (Integer): Perform gradients checking every N epochs [default: 25]
    def initialize(gradient_checks: :warning, nbr_gradient_checks_samples: 2, gradient_checks_epochs_interval: 25)
      @gradient_checks = gradient_checks
      @nbr_gradient_checks_samples = nbr_gradient_checks_samples
      @gradient_checks_epochs_interval = gradient_checks_epochs_interval
    end

    # Setup the gradients checker to perform on a given model and loss.
    #
    # Parameters::
    # * *model* (Model): Model for which gradient checking is done
    # * *loss* (Loss): Loss instance to be used
    def link_to_model(model, loss)
      @model = model
      @loss = loss
    end

    # Check gradients while performing gradient descent.
    #
    # Parameters::
    # * *idx_epoch* (Integer): Index of the epoch for which we perform gradient checking
    # * *minibatch* (RubyNeuralNets::Minibatch): The minibatch containing input and reference data
    # * Code: Code called to perform gradient descent on the model
    def check_gradients_for(idx_epoch, minibatch)
      # Compute d_theta_approx for gradient checking before modifying parameters with back propagation
      gradient_checking_epsilon = 1e-7
      d_theta_approx = nil
      parameters = nil
      perform_gradient_checking = @gradient_checks != :off &&
        idx_epoch % @gradient_checks_epochs_interval == 0 &&
        # Skip gradient checking if L2 regularization is used (they don't work well together)
        (!@loss.respond_to?(:weight_decay) || @loss.weight_decay == 0)

      if perform_gradient_checking
        minibatch_input = minibatch.input
        minibatch_target = minibatch.target
        m = minibatch.size
        parameters = @model.parameters
        d_theta_approx = Numo::DFloat[
          *parameters.map do |parameter|
            # Compute the indexes to select from the parameter
            parameter.gradient_check_indices = @nbr_gradient_checks_samples.times.map { rand(parameter.values.size) }.sort.uniq
            parameter.gradient_check_indices.map do |idx_param|
              value_original = parameter.values[idx_param]
              begin
                parameter.values[idx_param] = value_original - gradient_checking_epsilon
                cost_minus = @loss.compute_loss(@model.forward_propagate(minibatch_input), minibatch_target, @model).sum / m
                parameter.values[idx_param] = value_original + gradient_checking_epsilon
                cost_plus = @loss.compute_loss(@model.forward_propagate(minibatch_input), minibatch_target, @model).sum / m
                (cost_plus - cost_minus) / (2 * gradient_checking_epsilon)
              ensure
                parameter.values[idx_param] = value_original
              end
            end
          end.flatten(1)
        ]
      end

      # Call gradient descent
      yield

      if perform_gradient_checking
        # Compute d_theta for gradient checking
        d_theta = nil
        parameters.each do |parameter|
          dparams = parameter.dparams[parameter.gradient_check_indices]
          d_theta = d_theta.nil? ? dparams : d_theta.concatenate(dparams)
        end
        # Perform gradient checking
        gradient_distance = Helpers.norm_2(d_theta_approx - d_theta) / (Helpers.norm_2(d_theta_approx) + Helpers.norm_2(d_theta))
        log "Gradient checking on #{d_theta.size} parameters got #{gradient_distance}"
        if gradient_distance > gradient_checking_epsilon * 100
          # Debug breakdown per-parameter tensor to locate mismatch source
          offset = 0
          parameters.each_with_index do |parameter, idx_param_tensor|
            nbr_indices = parameter.gradient_check_indices.size
            num = d_theta_approx[offset...offset + nbr_indices]
            ana = d_theta[offset...offset + nbr_indices]
            dist = Helpers.norm_2(num - ana) / (Helpers.norm_2(num) + Helpers.norm_2(ana))
            log "  Param ##{idx_param_tensor} name=#{parameter.name} shape=#{parameter.shape.inspect} rel_dist=#{dist}"
            parameter.gradient_check_indices.each_with_index do |param_idx, i|
              log "    idx=#{param_idx} d_theta_approx=#{num[i]} d_theta=#{ana[i]}"
            end
            offset += nbr_indices
          end
          Helpers.handle_error("Gradient checking reports a distance of #{gradient_distance} for an epsilon of #{gradient_checking_epsilon}", @gradient_checks)
        end
      end
    end

  end

end
