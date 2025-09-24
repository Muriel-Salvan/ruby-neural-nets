module RubyNeuralNets

  # Track the progress of training models
  class ProgressTracker

    # Constructor
    #
    # Parameters::
    # * *display_graphs* (Boolean): Do we want to display graphs of loss and accuracy? [default: true]
    def initialize(display_graphs: true)
      @display_graphs = display_graphs
    end

    # Start tracking progress for a training.
    #
    # Parameters::
    # * *classes* (Array<String>): List of classes that the model output will classify
    # * *loss* (Loss): The loss instance to be used
    # * *accuracy* (Accuracy): The accuracy instance to be used
    # * Code: Code called with the progress tracker instantiated to be used
    def track(classes, loss, accuracy)
      @loss = loss
      @accuracy = accuracy
      # Initialize graphs
      if @display_graphs
        @costs = []
        @cost_graph = Numo::Gnuplot.new
        @cost_graph.set terminal: 'wxt 0 position 0,0 size 640,400'
        @cost_graph.set title: 'Cost'
        @accuracies = []
        @accuracy_graph = Numo::Gnuplot.new
        @accuracy_graph.set terminal: 'wxt 0 position 640,0 size 640,400'
        @accuracy_graph.set title: 'Accuracy'
        @confusion_graph = Numo::Gnuplot.new
        @confusion_graph.set terminal: 'wxt 0 position 1280,0 size 640,400'
        @confusion_graph.set title: 'Confusion Matrix'
        @confusion_graph.set palette: 'gray'
        @confusion_graph.set xlabel: 'Predicted'
        @confusion_graph.set ylabel: 'True'
        max_idx = (classes.size - 0.5)
        @confusion_graph.set xrange: -0.5..max_idx
        @confusion_graph.set yrange: max_idx..-0.5
        tics = "(#{classes.map.with_index { |class_name, idx| "\"#{class_name}\" #{idx}" }.join(', ')})"
        @confusion_graph.set xtics: tics
        @confusion_graph.set ytics: tics
      end

      yield

      # Close graphs
      if @display_graphs
        puts 'Wait for user to close graphs'
        [@cost_graph, @accuracy_graph, @confusion_graph].map do |gnuplot_graph|
          Thread.new { gnuplot_graph.pause 'mouse close' }
        end.each(&:join)
      end
    end

    # Track the progress of a minibatch training
    #
    # Parameters::
    # * *idx_epoch* (Integer): Epoch's index
    # * *idx_minibatch* (Integer): Minibatch index
    # * *minibatch_x* (Numo::DFloat): Minibatch input that has just be forward propagated
    # * *minibatch_y* (Numo::DFloat): Minibatch reference
    # * *a* (Numo::DFloat): Minibatch prediction, result of the forward propagation
    def progress(idx_epoch, idx_minibatch, minibatch_x, minibatch_y, a)
      cost = @loss.compute_loss(a, minibatch_y).sum / minibatch_x.shape[1]
      accuracy = @accuracy.measure(a, minibatch_y)
      puts "[ProgressTracker] - [Epoch #{idx_epoch} - Minibatch #{idx_minibatch}] - Cost #{cost}, Training accuracy #{accuracy * 100}%"

      # Update graphs
      if @display_graphs
        @costs << cost
        @cost_graph.plot @costs, w: 'lines', t: ''
        @accuracies << accuracy
        @accuracy_graph.plot @accuracies, w: 'lines', t: ''
        @confusion_graph.plot @accuracy.confusion_matrix(a, minibatch_y), w: 'image', t: ''
      end
    end

  end

end
