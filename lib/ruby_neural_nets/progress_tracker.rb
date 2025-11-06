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
        @graph_row_padding = 80
        # Width of the graphs, unless this width would prevent all graphs to fit on screen.
        @graph_default_width = 800
        @graph_aspect_ratio = 0.75
        @screen_width, @screen_height =
          begin
            if RUBY_PLATFORM.include?('mingw') || RUBY_PLATFORM.include?('mswin')
              # Windows
              [
                `wmic path Win32_VideoController get CurrentHorizontalResolution /value`.split('=')[1].strip.to_i,
                `wmic path Win32_VideoController get CurrentVerticalResolution /value`.split('=')[1].strip.to_i
              ]
            elsif RUBY_PLATFORM.include?('linux')
              # Linux
              [
                `xrandr --current | grep '*' | head -1 | awk -F'x' '{print $1}'`.strip.to_i,
                `xrandr --current | grep '*' | head -1 | awk -F'x' '{print $2}' | cut -d' ' -f1`.strip.to_i
              ]
            else
              # Other platforms or fallback
              [2560, 1600]
            end
          rescue
            [2560, 1600] # Fallback if detection fails
          end
        # Apply some margins (taskbar, windows...)
        @screen_height -= 128
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
          accuracies: [],
          colors: GNUPLOT_COLORS[@experiments.select { |exp_id, exp_data| exp_id != experiment.exp_id && exp_data[:experiment].dev_experiment.nil? }.size % GNUPLOT_COLORS.size]
        )
        # Create shared Cost and Accuracy graphs (reused across experiments)
        create_graph('Cost', key: ['below', font: ',7']) if @graphs['Cost'].nil?
        create_graph('Accuracy', key: ['below', font: ',7']) if @graphs['Accuracy'].nil?

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
    # * *minibatch* (RubyNeuralNets::Minibatch): Minibatch containing input and reference data
    # * *a* (Object): Minibatch prediction, result of the forward propagation
    # * *loss* (Object): Computed loss for the minibatch
    def progress(experiment, idx_epoch, idx_minibatch, minibatch, a, loss)
      cost = loss.mean.to_f
      accuracy = @experiments[experiment.exp_id][:experiment].accuracy.measure(a, minibatch)
      log "[Epoch #{idx_epoch}] [Exp #{experiment.exp_id}] [Minibatch #{idx_minibatch}] - Cost #{cost}, Accuracy #{accuracy * 100}%"

      # Update graphs
      if @display_graphs
        @experiments[experiment.exp_id][:costs] << cost
        graph_lines(@graphs['Cost'], :costs)
        @experiments[experiment.exp_id][:accuracies] << accuracy
        graph_lines(@graphs['Accuracy'], :accuracies)
        @graphs["Confusion Matrix #{experiment.exp_id}"].plot(
          @experiments[experiment.exp_id][:experiment].accuracy.confusion_matrix(a, minibatch),
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
          sample_dim = minibatch.x.shape.index(minibatch.size)
          raise "Unable to determine sample dimension for minibatch.x shape #{minibatch.x.shape} and minibatch.size #{minibatch.size}" if sample_dim.nil?

          slices = Array.new(minibatch.x.shape.size, nil)
          plot_bitmaps(
            @graphs["Samples #{experiment.exp_id}"],
            [experiment.display_samples, minibatch.size].min.times.map do |idx_sample|
              slices[sample_dim] = idx_sample
              # TODO: Add a method in Minibatch that will return the slice without guessing the dimension by comparing batch size.
              Numo::UInt8[*((minibatch.x[*slices].flatten * 255).round)].reshape(rows, cols, channels)
            end
          )
        end
      end
    end

    private

    # List of Gnuplot colors pairs to be used in graphs.
    # First color is a vivid one, second color is the same hue but darker.
    GNUPLOT_COLORS = [
      %w[blue midnight-blue],
      %w[red dark-red],
      %w[green dark-green],
      %w[magenta dark-violet],
      %w[gold dark-goldenrod],
      %w[orange dark-orange]
    ]

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
      graph.set title: title.gsub('_', '\_')
      # Set additional properties
      kwargs.each do |property, value|
        graph.set property => value
      end

      # Store in @graphs hash using title as key
      @graphs[title] = graph

      # Make sure all graphs fit in the screen
      # Compute the optimal graph width that would allow @graphs.size graphs of aspect ratio @graph_aspect_ratio to fit in a screen of dimensions @screen_width, @screen_height, also taking into account that each row of graphs is separated by @graph_row_padding

      # Find the min possible value of number of graphs per row by computing the max graph width that allow all pixels to fit in the screen pixels, without considering geometry constraints.
      # Formula comes from the following:
      # * max_graph_pixels = max_graph_width * ((max_graph_width * @graph_aspect_ratio).to_i + @graph_row_padding)
      # * nbr_graphs_max = @screen_width * @screen_height / max_graph_pixels
      # * We look for max_graph_width that ensures @graphs.size <= nbr_graphs_max
      # * This gives us max_graph_width = - @graph_row_padding + Math.sqrt(@graph_row_padding * @graph_row_padding + (4 * @graph_aspect_ratio * @screen_width * @screen_height) / @graphs.size.to_f) / (2 * @graph_aspect_ratio)
      # We therefore start by considering those number of graphs per row, and increase them till it fits using the geometry constraints.
      nbr_graphs_per_row = @screen_width / ((- @graph_row_padding + Math.sqrt(@graph_row_padding * @graph_row_padding + (4 * @graph_aspect_ratio * @screen_width * @screen_height) / @graphs.size.to_f) / (2 * @graph_aspect_ratio)).to_i)
      nbr_graphs_rows = nil
      loop do
        nbr_graphs_rows = (@graphs.size.to_f / nbr_graphs_per_row).ceil
        # Find the minimal width that would fit nbr_graphs_per_row on our screen without leaving place for an extra graph on the same row
        min_width = @screen_width / (nbr_graphs_per_row + 1) + 1
        min_height = (min_width * @graph_aspect_ratio).to_i
        break if nbr_graphs_rows * (min_height + @graph_row_padding) <= @screen_height
        # That doesn't fit: increase the number of graphs per row
        nbr_graphs_per_row += 1
      end
      # Here we know which number of graphs per row we can fit.
      # Compute the max width possible that makes sure it fits for both width and height.
      # We look for the max width that satisfies both conditions:
      # * nbr_graphs_per_row * width <= @screen_width
      # * (nbr_graphs_rows * ((width * @graph_aspect_ratio).to_i + @graph_row_padding) <= @screen_height
      # Then we cap it at @graph_default_width (we don't need huge graphs on screen)
      graph_width = [
        @screen_width / nbr_graphs_per_row,
        ((@screen_height / nbr_graphs_rows - @graph_row_padding) / @graph_aspect_ratio).to_i,
        @graph_default_width
      ].min

      # Apply those dimensions and place all graphs
      graph_height = (graph_width * @graph_aspect_ratio).to_i
      @graphs.each_with_index do |(_name, graph), graph_index|
        graph.set terminal: "wxt 0 position #{(graph_index % nbr_graphs_per_row) * graph_width},#{(graph_index / nbr_graphs_per_row) * (graph_height + @graph_row_padding)} size #{graph_width},#{graph_height}"
      end
    end

    # Graph lines of a given measure for all experiments on a GnuPlot graph
    #
    # Parameters::
    # * *gnuplot_graph* (Numo::Gnuplot): The GnuPlot graph to draw on
    # * *measure* (Symbol): The measure to be graphed
    def graph_lines(gnuplot_graph, measure)
      plot_data = []
      @experiments.select { |exp_id, exp_data| !exp_data[measure].empty? }.each do |exp_id, exp_data|
        experiment = exp_data[:experiment]
        nbr_minibatches = experiment.dataset.size
        x_values = (0...exp_data[measure].size).map { |i| i.to_f / nbr_minibatches }
        y_values = exp_data[measure]

        # Add line plot
        plot_data << [
          x_values,
          y_values,
          with: 'lines',
          title: exp_id.gsub('_', '\_'),
          linecolor: "rgb \"#{experiment.dev_experiment.nil? ? exp_data[:colors][1] : @experiments[experiment.dev_experiment.exp_id][:colors][0]}\"",
          linewidth: experiment.dev_experiment.nil? ? 1.5 : 1
        ]

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
