require 'ruby_neural_nets/helpers'
require 'ruby_neural_nets/logger'
require 'numo/random'

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
      rng = Random.new(dataset_seed)
      numo_rng = Numo::Random::Generator.new(seed: dataset_seed)

      @partitioned_dataset = new_partitioned_dataset(name: dataset, rng:, numo_rng:)
      # Build the set of batching datasets, per partition name.
      # 3 kinds of datasets are considered:
      # 1. Preprocessing: This dataset contains all deterministic transformations that are needed to
      #     prepare individual samples to be served to the model. This can be cached.
      # 2. Augmentation: This dataset contains all random transformations that are used for the training
      #     set only. Their purpose is to add more training data from the preprocessed samples. This
      #     should not be cached.
      # 3. Batching: This dataset is the final one, containing shuffling and minibatches creation. This
      #     should not be cached.
      # Hash< String, Dataset >
      @partition_datasets = @partitioned_dataset.partitions.to_h do |partition|
        partition_dataset = @partitioned_dataset.clone
        partition_dataset.select_partition(partition)
        preprocessed_dataset = new_preprocessing_dataset(partition_dataset)
        # Minibatches and data augmentation are only used for the training partition.
        [
          partition,
          new_batching_dataset(
            partition == :training ? new_augmentation_dataset(preprocessed_dataset, rng:, numo_rng:) : preprocessed_dataset,
            rng:,
            numo_rng:,
            max_minibatch_size: partition == :training ? max_minibatch_size : 1000000000
          )
        ]
      end
    end

    # Instantiate a partitioned dataset.
    #
    # Parameters::
    # * *name* (String): Dataset name containing real data
    # * *rng* (Random): The random number generator to be used
    # * *numo_rng* (Numo::Random::Generator): The Numo random number generator to be used
    # Result::
    # * LabeledDataPartitioner: The partitioned dataset.
    def new_partitioned_dataset(name:, rng:, numo_rng:)
      raise 'Not implemented'
    end

    # Return a preprocessing dataset for this data loader.
    #
    # Parameters::
    # * *dataset* (Dataset): The partitioned dataset
    # Result::
    # * Dataset: The dataset with preprocessing applied
    def new_preprocessing_dataset(dataset)
      raise 'Not implemented'
    end

    # Return an augmentation dataset for this data loader.
    # This is only used for the training dataset.
    #
    # Parameters::
    # * *preprocessed_dataset* (Dataset): The preprocessed dataset
    # * *rng* (Random): The random number generator to be used
    # * *numo_rng* (Numo::Random::Generator): The Numo random number generator to be used
    # Result::
    # * Dataset: The dataset with augmentation applied
    def new_augmentation_dataset(preprocessed_dataset, rng:, numo_rng:)
      raise 'Not implemented'
    end

    # Return a batching dataset for this data loader.
    #
    # Parameters::
    # * *augmented_dataset* (Dataset): The augmented dataset
    # * *rng* (Random): The random number generator to be used
    # * *numo_rng* (Numo::Random::Generator): The Numo random number generator to be used
    # * *max_minibatch_size* (Integer): The required minibatch size
    # Result::
    # * Dataset: The dataset with batching applied
    def new_batching_dataset(augmented_dataset, rng:, numo_rng:, max_minibatch_size:)
      raise 'Not implemented'
    end

    # Return the labels of the dataset.
    #
    # Result::
    # * Array< String >: List of labels of the dataset
    def labels
      @partition_datasets.values.first.labels
    end

    # Get the dataset of a given dataset type (training, dev, test...)
    #
    # Parameters::
    # * *dataset_type* (Symbol)
    # Result::
    # * Dataset: Corresponding dataset
    def dataset(dataset_type)
      @partition_datasets[dataset_type]
    end

    # Get some labels stats on the dataset
    #
    # Parameters::
    # * *dataset_type* (Symbol): Dataset type (:training, :dev, :test...)
    # Result:
    # * Hash<String, Hash>: Some statistics, per label. Here are the available statistics:
    #   * *nbr_elements* (Integer): Number of elements for this label
    def label_stats(dataset_type)
      @partitioned_dataset.select_partition(dataset_type) do
        labels.to_h do |label|
          [
            label,
            {
              nbr_elements: @partitioned_dataset.select { |(_x, select_label)| select_label == label }.size
            }
          ]
        end
      end
    end

    # Display some dataset statistics in the terminal
    def display_stats
      partitions = @partitioned_dataset.partitions
      known_labels = labels
      dataset_types_totals = partitions.to_h { |dataset_type| [dataset_type, 0] }
      dataset_types_stats = partitions.to_h { |dataset_type| [dataset_type, label_stats(dataset_type)] }
      require 'terminal-table'
      puts(
        Terminal::Table.new(
          title: "Dataset statistics (#{known_labels.size} labels)",
          headings: ['Label'] + partitions.map(&:to_s) + ['Total']
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
      @partitioned_dataset.select_partition(dataset_type) do
        loop do
          found_file, found_label = @partitioned_dataset[rand(@partitioned_dataset.size)]
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
      @partition_datasets.values.first.image_stats
    end

  end

end
