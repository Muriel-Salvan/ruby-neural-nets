require 'ruby_neural_nets/helpers'
require 'ruby_neural_nets/logger'

module RubyNeuralNets

  # Base class representing a data loader, serving data.
  # Technically it is a factory of Dataset.
  class DataLoader
    include Logger

    # Constructor
    #
    # Parameters::
    # * *dataset* (String): The dataset name
    # * *max_minibatch_size* (Integer): Max size each minibatch should have
    def initialize(dataset:, max_minibatch_size:, dataset_seed:)
      @dataset_name = dataset
      @max_minibatch_size = max_minibatch_size
      @dataset_rng = Random.new(dataset_seed)
      # This dataset is used to get statistics on partitions and labels
      @elements_labels_dataset = new_elements_labels_dataset
      # This dataset is used to access the model's inputs (X) and compare them with true outputs (Y)
      @elements_dataset = new_elements_dataset
    end

    # Create an elements dataset that maps to the labels and is partitioned.
    # The following instance variables can be used to initialize the dataset correctly:
    # * *@dataset_name* (String): The dataset name.
    #
    # Result::
    # * Dataset: The dataset that will serve data with Y being true labels, unencoded
    def new_elements_labels_dataset
      raise 'Not implemented'
    end

    # Create an elements dataset for this data loader.
    # The following instance variables can be used to initialize the dataset correctly:
    # * *@dataset_name* (String): The dataset name.
    # * *@elements_labels_dataset* (Dataset): The elements dataset previously created with true labels.
    #
    # Result::
    # * Dataset: The dataset that will serve data with X and Y being prepared for training
    def new_elements_dataset
      raise 'Not implemented'
    end

    # Return a minibatch dataset for this data loader.
    # The following instance variables can be used to initialize the dataset correctly:
    # * *@dataset_name* (String): The dataset name.
    # * *@elements_labels_dataset* (Dataset): The elements dataset previously created with true labels.
    # * *@elements_dataset* (Dataset): The elements dataset previously created.
    #
    # Parameters::
    # * *max_minibatch_size* (Integer): The required minibatch size
    # Result::
    # * Dataset: The dataset that will serve data as minibatches
    def new_minibatch_dataset(max_minibatch_size:)
      raise 'Not implemented'
    end

    # Return the labels of the dataset.
    #
    # Result::
    # * Array< String >: List of labels of the dataset
    def labels
      @elements_labels_dataset.labels
    end

    # Select a dataset type (training, dev, test...) for the data to be loaded
    #
    # Parameters::
    # * *dataset_type* (Symbol)
    # * Code: Code called with the partition selected
    def select_dataset_type(dataset_type)
      @elements_dataset.select_partition(dataset_type) do
        # Invalidate eventual caches
        @elements_dataset.invalidate
        yield
      end
    end

    # Loop over the data using minibatches
    #
    # Parameters::
    # * *max_minibatch_size* (Integer): The required minibatch size [defaults: @max_minibatch_size]
    # * Code: The code called for each minibatch
    #   * *minibatch_x* (Object): The X component of the minibatch (inputs)
    #   * *minibatch_y* (Object): The Y component of the minibatch (expected real outputs)
    #   * *minibatch_size* (Integer): The minibatch size
    def each_minibatch(max_minibatch_size: @max_minibatch_size)
      minibatch_dataset = new_minibatch_dataset(max_minibatch_size:)
      minibatch_dataset.prepare_for_epoch
      minibatch_dataset.each do |minibatch_x, (minibatch_y, minibatch_size)|
        yield minibatch_x, minibatch_y, minibatch_size
      end
    end

    # Get some labels stats on the dataset
    #
    # Parameters::
    # * *dataset_type* (Symbol): Dataset type (:training, :dev, :test...)
    # Result:
    # * Hash<String, Hash>: Some statistics, per label. Here are the available statistics:
    #   * *nbr_elements* (Integer): Number of elements for this label
    def label_stats(dataset_type)
      @elements_labels_dataset.select_partition(dataset_type) do
        labels.to_h do |label|
          [
            label,
            {
              nbr_elements: @elements_labels_dataset.select { |(_x, select_label)| select_label == label }.size
            }
          ]
        end
      end
    end

    # Display some dataset statistics in the terminal
    def display_stats
      partitions = @elements_labels_dataset.partitions
      known_labels = labels
      dataset_types_totals = partitions.to_h { |dataset_type| [dataset_type, 0] }
      dataset_types_stats = partitions.to_h { |dataset_type| [dataset_type, label_stats(dataset_type)] }
      require 'terminal-table'
      puts(
        Terminal::Table.new(
          title: "Dataset statistics (#{known_labels.size} labels)",
          headings: ['Class'] + partitions.map(&:to_s) + ['Total']
        ) do |t|
          known_labels.each do |label|
            dataset_files = partitions.map do |dataset_type|
              nbr_files = dataset_types_stats[dataset_type][label][:nbr_elements]
              dataset_types_totals[dataset_type] += nbr_files
              nbr_files
            end
            t << [label] + dataset_files + [dataset_files.sum]
          end
          t.add_separator
          total = dataset_types_totals.values.sum
          t << ['Total'] + partitions.map { |dataset_type| dataset_types_totals[dataset_type] } + [total]
          t << [''] + partitions.map { |dataset_type| "#{(dataset_types_totals[dataset_type] * 100) / total}%" } + ['100%']
        end
      )
    end

    # Display a sample image from a dataset
    #
    # Parameters::
    # * *dataset_type* (Symbol): Dataset type from which the sample should be taken
    # * *label* (String): Label from which the sample should be taken
    def display_sample(dataset_type, label)
      @elements_labels_dataset.select_partition(dataset_type) do
        loop do
          found_file, found_label = @elements_labels_dataset[rand(@elements_labels_dataset.size)]
          if found_label == label
            log "Display sample image #{found_file} of label #{found_label}"
            Helpers.display_image(Magick::ImageList.new(found_file).first)
            break
          end
        end
      end
    end

    # Get some images stats.
    # Those are supposed to be the same for all samples from the dataset and can be used to compute the model's architecture.
    #
    # Result::
    # * Hash: Image stats:
    #   * *rows* (Integer): Number of rows
    #   * *cols* (Integer): Number of columns
    #   * *channels* (Integer): Number of channels
    def image_stats
      @elements_dataset.image_stats
    end

  end

end
