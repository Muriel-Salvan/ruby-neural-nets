require 'ruby_neural_nets/logger'

module RubyNeuralNets

  # Track the progress of training models
  class ProgressTracker
    include Logger

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
        @screen_width = 1280
        @graph_width = 640
        @graph_height = 400
        @graph_row_padding = 80
        # Set of named graphs
        # Hash< String, Numo::GnuPlot >
        @graphs = {}
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
        create_graph('Cost')
        @accuracies = []
        create_graph('Accuracy')
        max_idx = (classes.size - 0.5)
        tics = "(#{classes.map.with_index { |class_name, idx| "\"#{class_name}\" #{idx}" }.join(', ')}) "
        create_graph(
          'Confusion Matrix',
          palette: 'gray',
          xlabel: 'Predicted',
          ylabel: 'True',
          xrange: -0.5..max_idx,
          yrange: max_idx..-0.5,
          xtics: tics,
          ytics: tics
        )
        unless @display_units.empty?
          # Change it to resolve the exact parameters selected
          # Array< [ Parameter, Array< Array< Integer >           > ] >
          # Array< [ parameter,        unit_indexes_per_graph_row   ] >
          @display_units = @display_units.map.with_index do |(name, nbr_units), idx_param|
            name = name.to_s if name.is_a?(Symbol)
            found_param = @model.parameters(name:).first
            raise "Unable to find parameter #{name} for plotting. Parameters are #{@model.parameters.map(&:name).join(', ')}" if found_param.nil?
            create_graph(found_param.name.gsub('_', '\_'))
            units_step = found_param.shape[0].to_f / nbr_units
            units_to_plot = nbr_units.times.map { |idx_unit| Integer(idx_unit * units_step) }.uniq
            [
              found_param,
              units_to_plot.each_slice(Math.sqrt(units_to_plot.size).round).to_a
            ]
          end
        end
      end

      yield

      # Close graphs
      if @display_graphs
        log 'Wait for user to close graphs'
        graphs_to_close = @graphs.values
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
      log "[Epoch #{idx_epoch} - Minibatch #{idx_minibatch}] - Cost #{cost}, Training accuracy #{accuracy * 100}%"

      # Update graphs
      if @display_graphs
        @costs << cost
        @graphs['Cost'].plot @costs, with: 'lines', title: ''
        @accuracies << accuracy
        @graphs['Accuracy'].plot @accuracies, with: 'lines', title: ''
        @graphs['Confusion Matrix'].plot @accuracy.confusion_matrix(a, minibatch_y, minibatch_size), with: 'image', title: ''
        unless @display_units.empty?
          @display_units.each do |(param, row_indices)|
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
            @graphs[param.name.gsub('_', '\_')].plot param_img.reverse(0), with: 'rgbimage', title: ''
          end
        end
      end
    end

    private

    # Create a new graph with automatic positioning
    #
    # Parameters:::
    # * *title* (String): The title of the graph
    # * *kwargs: GnuPlot properties to set as keyword arguments
    def create_graph(title, **kwargs)
      graph = Numo::Gnuplot.new
      # Calculate position based on number of existing graphs
      graph_index = @graphs.size
      x_pos = (graph_index * @graph_width) % @screen_width
      y_pos = ((graph_index * @graph_width) / @screen_width).floor * (@graph_height + @graph_row_padding)

      graph.set terminal: "wxt 0 position #{x_pos},#{y_pos} size #{@graph_width},#{@graph_height}"
      graph.set title: title

      # Set additional properties
      kwargs.each do |property, value|
        graph.set property => value
      end

      # Store in @graphs hash using title as key
      @graphs[title] = graph
      nil
    end

  end

end
