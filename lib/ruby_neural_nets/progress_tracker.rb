require 'ruby_neural_nets/logger'

module RubyNeuralNets

  # Track the progress of training and evaluating models
  class ProgressTracker
    include Logger

    # Constructor
    #
    # Parameters::
    # * *display_graphs* (Boolean): Do we want to display graphs ? [default: true]
    def initialize(display_graphs: true)
      @display_graphs = display_graphs
      if @display_graphs
        require 'numo/gnuplot'
        require 'numo/narray'
        @screen_width = 2560
        @graph_width = 640
        @graph_height = 400
        @graph_row_padding = 80
        # Set of named graphs
        # Hash< String, Numo::GnuPlot >
        @graphs = {}
      end
      # Hash to store experiment data with experiment IDs as keys
      # Hash< String, Hash >
      @experiments = {}
    end

    # Start tracking progress for a training.
    #
    # Parameters::
    # * *experiment* (Experiment): The experiment object to track progress for
    def track(experiment)
      @experiments[experiment.exp_id] = {
        experiment: experiment,
        early_stopping_epoch: nil
      }
      # Initialize graphs
      if @display_graphs
        @experiments[experiment.exp_id].merge!(
          costs: [],
          accuracies: []
        )
        # Create shared Cost and Accuracy graphs (reused across experiments)
        create_graph('Cost') if @graphs['Cost'].nil?
        create_graph('Accuracy') if @graphs['Accuracy'].nil?

        # Create experiment-specific graphs for Confusion Matrix and display units
        labels = experiment.dataset.labels
        max_idx = (labels.size - 0.5)
        tics = "(#{labels.map.with_index { |class_name, idx| "\"#{class_name}\" #{idx}" }.join(', ')}) "
        create_graph(
          "Confusion Matrix #{experiment.exp_id}",
          palette: 'gray',
          xlabel: 'Predicted',
          ylabel: 'True',
          xrange: -0.5..max_idx,
          yrange: max_idx..-0.5,
          xtics: tics,
          ytics: tics
        )
        unless experiment.display_units.empty?
          # Change it to resolve the exact parameters selected
          # Array< [ Parameter, Array< Array< Integer >           > ] >
          # Array< [ parameter,        unit_indexes_per_graph_row   ] >
          @experiments[experiment.exp_id][:display_units] = experiment.display_units.map.with_index do |(name, nbr_units), idx_param|
            name = name.to_s if name.is_a?(Symbol)
            found_param = experiment.model.parameters(name:).first
            raise "Unable to find parameter #{name} for plotting. Parameters are #{experiment.model.parameters.map(&:name).join(', ')}" if found_param.nil?
            create_graph("#{found_param.name} #{experiment.exp_id}")
            units_step = found_param.shape[0].to_f / nbr_units
            units_to_plot = nbr_units.times.map { |idx_unit| Integer(idx_unit * units_step) }.uniq
            [
              found_param,
              units_to_plot.each_slice(Math.sqrt(units_to_plot.size).round).to_a
            ]
          end
        end

        # Initialize display samples if specified
        create_graph("Samples #{experiment.exp_id}") if experiment.display_samples > 0
      end
    end

    # Close all graphs and wait for user to close them
    def close_graphs
      if @display_graphs
        log 'Wait for user to close graphs'
        graphs_to_close = @graphs.values
        graphs_to_close.map do |gnuplot_graph|
          Thread.new { gnuplot_graph.pause 'mouse close' }
        end.each(&:join)
      end
    end

    # Notify that early stopping has occurred for a training experiment
    #
    # Parameters::
    # * *training_experiment* (Experiment): The training experiment that reached early stopping
    # * *epoch* (Integer): The epoch at which early stopping occurred
    def notify_early_stopping(training_experiment, epoch)
      @experiments[training_experiment.exp_id][:early_stopping_epoch] = epoch
      log "Early stopping notified for [Exp #{training_experiment.exp_id}] at epoch #{epoch}"
    end

    # Track the progress of a minibatch training
    #
    # Parameters::
    # * *experiment* (Experiment): The experiment object to track progress for
    # * *idx_epoch* (Integer): Epoch's index
    # * *idx_minibatch* (Integer): Minibatch index
    # * *minibatch_x* (Object): Minibatch input that has just be forward propagated
    # * *minibatch_y* (Object): Minibatch reference
    # * *a* (Object): Minibatch prediction, result of the forward propagation
    # * *loss* (Object): Computed loss for the minibatch
    # * *minibatch_size* (Integer): Minibatch size
    def progress(experiment, idx_epoch, idx_minibatch, minibatch_x, minibatch_y, a, loss, minibatch_size)
      cost = loss.mean.to_f
      accuracy = @experiments[experiment.exp_id][:experiment].accuracy.measure(a, minibatch_y, minibatch_size)
      log "[Epoch #{idx_epoch}] [Exp #{experiment.exp_id}] [Minibatch #{idx_minibatch}] - Cost #{cost}, Accuracy #{accuracy * 100}%"

      # Update graphs
      if @display_graphs
        @experiments[experiment.exp_id][:costs] << cost
        graph_lines(@graphs['Cost'], :costs)
        @experiments[experiment.exp_id][:accuracies] << accuracy
        graph_lines(@graphs['Accuracy'], :accuracies)
        @graphs["Confusion Matrix #{experiment.exp_id}"].plot(
          @experiments[experiment.exp_id][:experiment].accuracy.confusion_matrix(a, minibatch_y, minibatch_size),
          with: 'image',
          title: ''
        )
        unless @experiments[experiment.exp_id][:display_units].nil?
          @experiments[experiment.exp_id][:display_units].each do |(param, row_indices)|
            tensor_size = param.shape[1]
            values = param.values
            nbr_channels = tensor_size % 3 == 0 ? 3 : 1
            width_float = Math.sqrt(tensor_size / nbr_channels)
            width = width_float.floor == width_float ? width_float.floor : width_float.floor + 1
            padding = width * width - tensor_size / nbr_channels
            bitmaps = []
            row_indices.flatten.each do |idx_unit|
              normalized_image = values[idx_unit, nil]
              min_value = normalized_image.min
              unit_img = Numo::UInt8[*(((normalized_image - min_value) / (normalized_image.max - min_value)) * 255).round].
                concatenate(Numo::UInt8.zeros(padding * nbr_channels)).
                reshape(width, width, nbr_channels)
              bitmaps << unit_img
            end
            plot_bitmaps(@graphs["#{param.name} #{experiment.exp_id}"], bitmaps)
          end
        end

        # Display samples
        if experiment.display_samples > 0
          image_stats = experiment.data_loader.image_stats
          rows = image_stats[:rows]
          cols = image_stats[:cols]
          channels = image_stats[:channels]
          # Detect the sample dimension based on minibatch_size
          sample_dim = minibatch_x.shape.index(minibatch_size)
          raise "Unable to determine sample dimension for minibatch_x shape #{minibatch_x.shape} and minibatch_size #{minibatch_size}" if sample_dim.nil?

          slices = Array.new(minibatch_x.shape.size, nil)
          plot_bitmaps(
            @graphs["Samples #{experiment.exp_id}"],
            [experiment.display_samples, minibatch_size].min.times.map do |idx_sample|
              slices[sample_dim] = idx_sample
              Numo::UInt8[*((minibatch_x[*slices].flatten * 255).round)].reshape(rows, cols, channels)
            end
          )
        end
      end
    end

    private

    # Plot bitmaps on a given graph, stacking them horizontally and vertically
    #
    # Parameters:::
    # * *gnuplot_graph* (Numo::Gnuplot): The graph to plot on
    # * *bitmaps* (Array<Array<Array<Numo::UInt8>>>): Array of bitmaps, each bitmap is [rows, cols, channels]
    def plot_bitmaps(gnuplot_graph, bitmaps)
      # Group by rows
      rows_per_group = Math.sqrt(bitmaps.size).ceil.to_i
      max_width = 0
      row_imgs = bitmaps.each_slice(rows_per_group).map do |row_bitmaps|
        row_img = nil
        row_bitmaps.each do |bitmap|
          nbr_channels = bitmap.shape[2]
          # Ensure RGB
          bitmap_rgb = if nbr_channels == 1
                         bitmap.concatenate(bitmap, axis: 2).concatenate(bitmap, axis: 2)
                       else
                         bitmap
                       end
          row_img = row_img.nil? ? bitmap_rgb : row_img.concatenate(bitmap_rgb, axis: 1)
        end
        max_width = [max_width, row_img.shape[1]].max
        row_img
      end

      # Pad rows to max_width
      height = row_imgs.first.shape[0]
      channels = row_imgs.first.shape[2]

      gnuplot_graph.plot(
        row_imgs.
          map do |row_img|
            if row_img.shape[1] < max_width
              padding = Numo::UInt8.zeros(height, max_width - row_img.shape[1], channels) + 255
              row_img.concatenate(padding, axis: 1)
            else
              row_img
            end
          end.
          # Concatenate vertically
          reduce { |accum, row| accum.concatenate(row) }.
          reverse(0),
        with: 'rgbimage',
        title: ''
      )
    end

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
      graph.set title: title.gsub('_', '\_')

      # Set additional properties
      kwargs.each do |property, value|
        graph.set property => value
      end

      # Store in @graphs hash using title as key
      @graphs[title] = graph
    end

    # Graph lines of a given measure for all experiments on a GnuPlot graph
    #
    # Parameters::
    # * *gnuplot_graph* (Numo::Gnuplot): The GnuPlot graph to draw on
    # * *measure* (Symbol): The measure to be graphed
    def graph_lines(gnuplot_graph, measure)
      plot_data = []
      @experiments.select { |exp_id, exp_data| !exp_data[measure].empty? }.each do |exp_id, exp_data|
        nbr_minibatches = exp_data[:experiment].dataset.size
        x_values = (0...exp_data[measure].size).map { |i| i.to_f / nbr_minibatches }
        y_values = exp_data[measure]

        # Add line plot
        plot_data << [x_values, y_values, with: 'lines', title: exp_id.gsub('_', '\_')]

        # Add point for early stopping if applicable
        if exp_data[:early_stopping_epoch]
          index = exp_data[:early_stopping_epoch] * nbr_minibatches
          plot_data << [[x_values[index]], [y_values[index]], with: 'points pt 7 ps 1 lc rgb "red"', title: '']
        end
      end
      gnuplot_graph.plot(*plot_data)
    end

  end

end
