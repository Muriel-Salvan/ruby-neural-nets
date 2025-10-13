module RubyNeuralNets

  # Track the progress of training models
  class ProgressTracker

    # Constructor
    #
    # Parameters::
    # * *display_graphs* (Boolean): Do we want to display graphs ? [default: true]
    # * *display_units* (Hash<Symbol, String or Regexp, Integer>): For each parameter name (or regexp matching name), indicate the number of units we want to picture [default: {}]
    def initialize(display_graphs: true, display_units: {})
      @display_graphs = display_graphs
      @display_units = display_units
      if @display_graphs
        require 'numo/gnuplot'
        require 'numo/narray'
      end
    end

    # Start tracking progress for a training.
    #
    # Parameters::
    # * *model* (Model): The model that we are tracking
    # * *classes* (Array<String>): List of classes that the model output will classify
    # * *loss* (Loss): The loss instance to be used
    # * *accuracy* (Accuracy): The accuracy instance to be used
    # * Code: Code called with the progress tracker instantiated to be used
    def track(model, classes, loss, accuracy)
      @model = model
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
        unless @display_units.empty?
          # Change it to resolve the exact parameters selected
          # Array< [ Parameter, Numo::Gnuplot, Array< Array< Integer             > > ] >
          # Array< [ parameter, graph,                unit_indexes_per_graph_row   > ] >
          @display_units = @display_units.map.with_index do |(name, nbr_units), idx_param|
            name = name.to_s if name.is_a?(Symbol)
            found_param = @model.parameters(name:).first
            raise "Unable to find parameter #{name} for plotting. Parameters are #{@model.parameters.map(&:name).join(', ')}" if found_param.nil?
            parameter_graph = Numo::Gnuplot.new
            parameter_graph.set terminal: "wxt 0 position #{idx_param * 640},480 size 640,400"
            parameter_graph.set title: found_param.name.gsub('_', '\_')
            units_step = found_param.shape[0].to_f / nbr_units
            units_to_plot = nbr_units.times.map { |idx_unit| Integer(idx_unit * units_step) }.uniq
            [
              found_param,
              parameter_graph,
              units_to_plot.each_slice(Math.sqrt(units_to_plot.size).round).to_a
            ]
          end
        end
      end

      yield

      # Close graphs
      if @display_graphs
        puts 'Wait for user to close graphs'
        graphs_to_close = [@cost_graph, @accuracy_graph, @confusion_graph]
        graphs_to_close.concat(@display_units.map { |(_param, graph, _row_indices)| graph }) unless @display_units.empty?
        graphs_to_close.map do |gnuplot_graph|
          Thread.new { gnuplot_graph.pause 'mouse close' }
        end.each(&:join)
      end
    end

    # Track the progress of a minibatch training
    #
    # Parameters::
    # * *idx_epoch* (Integer): Epoch's index
    # * *idx_minibatch* (Integer): Minibatch index
    # * *minibatch_x* (Object): Minibatch input that has just be forward propagated
    # * *minibatch_y* (Object): Minibatch reference
    # * *a* (Object): Minibatch prediction, result of the forward propagation
    # * *loss* (Object): Computed loss for the minibatch
    # * *minibatch_size* (Integer): Minibatch size
    def progress(idx_epoch, idx_minibatch, minibatch_x, minibatch_y, a, loss, minibatch_size)
      cost = loss.mean.to_f
      accuracy = @accuracy.measure(a, minibatch_y, minibatch_size)
      puts "[ProgressTracker] - [Epoch #{idx_epoch} - Minibatch #{idx_minibatch}] - Cost #{cost}, Training accuracy #{accuracy * 100}%"

      # Update graphs
      if @display_graphs
        @costs << cost
        @cost_graph.plot @costs, with: 'lines', title: ''
        @accuracies << accuracy
        @accuracy_graph.plot @accuracies, with: 'lines', title: ''
        @confusion_graph.plot @accuracy.confusion_matrix(a, minibatch_y, minibatch_size), with: 'image', title: ''
        unless @display_units.empty?
          @display_units.each do |(param, graph, row_indices)|
            tensor_size = param.shape[1]
            values = param.values
            # Consider it to be RGB image if we can divide the number of input values by 3
            nbr_channels = tensor_size % 3 == 0 ? 3 : 1
            width_float = Math.sqrt(tensor_size / nbr_channels)
            width = width_float.floor == width_float ? width_float.floor : width_float.floor + 1
            # Compute possible padding needed to consider it a square image
            padding = width * width - tensor_size / nbr_channels
            param_img = nil
            row_indices.each do |indices|
              row_img = nil
              indices.each do |idx_unit|
                normalized_image = values[idx_unit, nil]
                min_value = normalized_image.min
                unit_img = Numo::UInt8[*(((normalized_image - min_value) / (normalized_image.max - min_value)) * 255).round].
                  concatenate(Numo::UInt8.zeros(padding * nbr_channels)).
                  reshape(width, width, nbr_channels)
                row_img = row_img.nil? ? unit_img : row_img.concatenate(unit_img, axis: 1)
              end
            param_img =
              if param_img.nil?
                row_img
              else
                # Pad row_img with white pixels on the right to match param_img width
                row_img = row_img.concatenate(Numo::UInt8.zeros(width, param_img.shape[1] - row_img.shape[1], nbr_channels) + 255, axis: 1) if row_img.shape[1] < param_img.shape[1]
                param_img.concatenate(row_img)
              end
            end
            # Duplicate all channels in param_img to make it RGB
            param_img = param_img.concatenate(param_img, axis: 2).concatenate(param_img, axis: 2) if nbr_channels == 1
            graph.plot param_img.reverse(0), with: 'rgbimage', title: ''
          end
        end
      end
    end

  end

end
