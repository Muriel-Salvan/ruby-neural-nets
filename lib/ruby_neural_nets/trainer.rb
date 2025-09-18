require 'ruby_neural_nets/accuracy'
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
    def initialize(
      nbr_epochs:,
      max_minibatch_size:,
      accuracy: Accuracy.new,
      loss: Losses::CrossEntropy.new,
      optimizer: Optimizers::Constant.new(learning_rate: 0.001)
    )
      @nbr_epochs = nbr_epochs
      @max_minibatch_size = max_minibatch_size
      @accuracy = accuracy
      @loss = loss
      @optimizer = optimizer
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
      losses = []
      accuracies = []

      loss_graph = nil
      accuracy_graph = nil
      confusion_graph = nil
      if display_graphs
        loss_graph = Numo::Gnuplot.new
        loss_graph.set terminal: 'wxt 0 position 0,0 size 640,400'
        loss_graph.set title: 'Loss'
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
          m = minibatch_x.shape[1]

          # Forward propagation
          a = model.forward_propagate(minibatch_x)

          # Compute loss and accuracy
          loss = @loss.compute_loss(a, minibatch_y) / m
          accuracy = @accuracy.measure(a, minibatch_y)
          puts "[Trainer] - [Epoch #{idx_epoch} - Minibatch #{idx_minibatch}] - Loss #{loss}, Training accuracy #{accuracy * 100}%"

          if display_graphs
            losses << loss
            loss_graph.plot losses, w: 'lines', t: ''
            accuracies << accuracy
            accuracy_graph.plot accuracies, w: 'lines', t: ''
            confusion_graph.plot @accuracy.confusion_matrix(a, minibatch_y), w: 'image', t: ''
          end

          # Gradient descent
          model.gradient_descent(@loss.compute_loss_gradient(a, minibatch_y), a, minibatch_y)

          idx_minibatch += 1
        end
      end

      if display_graphs
        puts 'Wait for user to close graphs'
        [loss_graph, accuracy_graph, confusion_graph].map do |gnuplot_graph|
          Thread.new { gnuplot_graph.pause 'mouse close' }
        end.each(&:join)
      end
    end

  end

end
