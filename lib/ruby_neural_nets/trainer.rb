require 'ruby_neural_nets/accuracy'
require 'ruby_neural_nets/helpers'
require 'ruby_neural_nets/losses/cross_entropy'

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
    # * *gradient_checks* (Symbol): Behavior when wrong gradient checking is detected [default: :warning]
    #   Possible values are the same as for Helpers.instability_checks
    # * *nbr_gradient_checks_samples* (Integer): Max number of parameters per model's parameter tensor to be used for gradient checking [default: 2]
    def initialize(
      nbr_epochs:,
      max_minibatch_size:,
      accuracy: Accuracy.new,
      loss: Losses::CrossEntropy.new,
      optimizer: Optimizers::Constant.new(learning_rate: 0.001),
      gradient_checks: :warning,
      nbr_gradient_checks_samples: 2
    )
      @nbr_epochs = nbr_epochs
      @max_minibatch_size = max_minibatch_size
      @accuracy = accuracy
      @loss = loss
      @optimizer = optimizer
      @gradient_checks = gradient_checks
      @nbr_gradient_checks_samples = nbr_gradient_checks_samples
    end

    # Train a given model on a training dataset
    #
    # Parameters::
    # * *model* (Model): The model to be trained
    # * *dataset* (Dataset): The dataset on which the model has to be trained
    # * *dataset_type* (Symbol): The dataset type on which the model has to be trained [default: :train]
    # * *display_graphs* (Boolean): Do we want to display graphs of loss and accuracy at the end? [default: true]
    def train(model, dataset, dataset_type: :train, display_graphs: true)
      puts "[Trainer] - Train with minibatches of size #{@max_minibatch_size}, on #{@nbr_epochs} epochs"
      costs = []
      accuracies = []

      cost_graph = nil
      accuracy_graph = nil
      confusion_graph = nil
      if display_graphs
        cost_graph = Numo::Gnuplot.new
        cost_graph.set terminal: 'wxt 0 position 0,0 size 640,400'
        cost_graph.set title: 'Cost'
        accuracy_graph = Numo::Gnuplot.new
        accuracy_graph.set terminal: 'wxt 0 position 640,0 size 640,400'
        accuracy_graph.set title: 'Accuracy'
        confusion_graph = Numo::Gnuplot.new
        confusion_graph.set terminal: 'wxt 0 position 1280,0 size 640,400'
        confusion_graph.set title: 'Confusion Matrix'
        confusion_graph.set palette: 'gray'
        confusion_graph.set xlabel: 'Predicted'
        confusion_graph.set ylabel: 'True'
        max_idx = (dataset.classes.size - 0.5)
        confusion_graph.set xrange: -0.5..max_idx
        confusion_graph.set yrange: max_idx..-0.5
        tics = "(#{dataset.classes.map.with_index { |class_name, idx| "\"#{class_name}\" #{idx}" }.join(', ')})"
        confusion_graph.set xtics: tics
        confusion_graph.set ytics: tics
      end

      @nbr_epochs.times do |idx_epoch|
        puts "[Trainer] - Training for epoch ##{idx_epoch}..."
        @optimizer.start_epoch(idx_epoch)
        idx_minibatch = 0
        dataset.for_each_minibatch(dataset_type, @max_minibatch_size) do |minibatch_x, minibatch_y|
          # Compute loss and accuracy
          cost, a, back_propagation_cache = minibatch_cost(model, minibatch_x, minibatch_y)
          accuracy = @accuracy.measure(a, minibatch_y)
          puts "[Trainer] - [Epoch #{idx_epoch} - Minibatch #{idx_minibatch}] - Cost #{cost}, Training accuracy #{accuracy * 100}%"

          if display_graphs
            costs << cost
            cost_graph.plot costs, w: 'lines', t: ''
            accuracies << accuracy
            accuracy_graph.plot accuracies, w: 'lines', t: ''
            confusion_graph.plot @accuracy.confusion_matrix(a, minibatch_y), w: 'image', t: ''
          end

          # Compute d_theta_approx for gradient checking before modifying parameters with back propagation
          gradient_checking_epsilon = 1e-7
          d_theta_approx = nil
          if @gradient_checks != :off
            d_theta_approx = Numo::DFloat[
              *model.parameters.map do |parameter|
                # Compute the indexes to select from the parameter
                parameter.gradient_check_indices = @nbr_gradient_checks_samples.times.map { rand(parameter.values.size) }.sort.uniq
                parameter.gradient_check_indices.map do |idx_param|
                  value_original = parameter.values[idx_param]
                  begin
                    parameter.values[idx_param] = value_original - gradient_checking_epsilon
                    cost_minus, _a, _cache = minibatch_cost(model, minibatch_x, minibatch_y)
                    parameter.values[idx_param] = value_original + gradient_checking_epsilon
                    cost_plus, _a, _cache = minibatch_cost(model, minibatch_x, minibatch_y)
                    (cost_plus - cost_minus) / (2 * gradient_checking_epsilon)
                  ensure
                    parameter.values[idx_param] = value_original
                  end
                end
              end.flatten(1)
            ]
          end

          # Gradient descent
          # Make sure gradient descent uses caches computed by the normal forward propagation
          model.back_propagation_cache = back_propagation_cache
          # TODO: Uncomment when OneLayer will work correctly
          # model.gradient_descent(@loss.compute_loss_gradient(a, minibatch_y) / minibatch_x.shape[1], a, minibatch_y)
          model.gradient_descent(a, minibatch_y)

          if @gradient_checks != :off
            # Compute d_theta for gradient checking
            d_theta = nil
            model.parameters.map do |parameter|
              dparams = parameter.dparams[parameter.gradient_check_indices]
              if d_theta.nil?
                d_theta = dparams
              else
                d_theta = d_theta.concatenate(dparams)
              end
            end
            # Perform gradient checking
            gradient_distance = Helpers.norm_2(d_theta_approx - d_theta) / (Helpers.norm_2(d_theta_approx) + Helpers.norm_2(d_theta))
            puts "[Trainer] - Gradient checking on #{d_theta.size} parameters got #{gradient_distance}"
            Helpers.handle_error("Gradient checking reports a distance of #{gradient_distance} for an epsilon of #{gradient_checking_epsilon}", @gradient_checks) if gradient_distance > gradient_checking_epsilon * 100
          end

          idx_minibatch += 1
        end
      end

      if display_graphs
        puts 'Wait for user to close graphs'
        [cost_graph, accuracy_graph, confusion_graph].map do |gnuplot_graph|
          Thread.new { gnuplot_graph.pause 'mouse close' }
        end.each(&:join)
      end
    end

    private

    # Compute cost from an input minibatch and a true result minibatch
    #
    # Parameters::
    # * *model* (Model): Model used to compute the cost
    # * *minibath_x* (Numo::DFloat): The input minibatch
    # * *minibath_y* (Numo::DFloat): The true output minibatch
    # Result::
    # * Float: Corresponding cost
    # * Numo::DFloat: Output of the model
    # * Hash: Cache of computations for back-propagation
    def minibatch_cost(model, minibatch_x, minibatch_y)
      a = model.forward_propagate(minibatch_x)
      [@loss.compute_loss(a, minibatch_y).sum / minibatch_x.shape[1], a, model.back_propagation_cache]
    end

  end

end
