require 'rmagick'
require 'numo/narray'

module RubyNeuralNets

  # Object representing a dataset, serving data
  class Dataset

    # Get the ordered list of classes
    #   Array<String>
    attr_reader :classes

    # Get the possible dataset types
    #   Array<Symbol>
    attr_reader :types

    # Constructor
    #
    # Parameters::
    # * *dataset_name* (String): Name of the dataset to read (sub-directory of the dataset directory)
    # * *percent_training* (Float): Percentage of the dataset to be used for training [default: 0.7]
    # * *percent_dev* (Float): Percentage of the dataset to be used for dev [default: 0.15]
    def initialize(dataset_name, percent_training: 0.7, percent_dev: 0.15)
      @types = %i[train dev test]
      @classes = Dir.glob("dataset/#{dataset_name}/*").
        select { |file| File.directory?(file) }.
        map { |file| File.basename(file) }.
        sort
      raise "No data in dataset #{dataset_name}" if @classes.empty?
      randomized_dataset = @classes.map do |class_name|
        Dir.glob("dataset/#{dataset_name}/#{class_name}/*.png").map do |file|
          [
            file,
            class_name
          ]
        end
      end.flatten(1).shuffle
      last_idx_train = (randomized_dataset.size * percent_training).to_i
      last_idx_dev = last_idx_train + (randomized_dataset.size * percent_dev).to_i
      @datasets = {
        train: randomized_dataset[0..last_idx_train],
        dev: randomized_dataset[last_idx_train + 1..last_idx_dev],
        test: randomized_dataset[last_idx_dev + 1..-1]
      }
      nbr_classes = @classes.size
      @class_one_hot = Hash[@classes.map.with_index do |class_name, idx_class|
        one_hot_vector = [0] * nbr_classes
        one_hot_vector[idx_class] = 1
        [
          class_name,
          one_hot_vector
        ]
      end]
      @minibatch_cache = []
    end

    # Loop over the dataset using minibatches
    #
    # Parameters::
    # * *dataset_type* (Symbol): Dataset type to loop over (can be :train, :dev or :test)
    # * *max_minibatch_size* (Integer): Max size each minibatch should have
    # * Code: Code called for each minibatch
    #   * *minibatch_x* (Numo::DFloat): Read minibatch X
    #   * *minibatch_y* (Numo::DFloat): Read minibatch Y
    def for_each_minibatch(dataset_type, max_minibatch_size)
      remaining_dataset = @datasets[dataset_type].dup
      idx_minibatch = 0
      while !remaining_dataset.empty?
        minibatch = remaining_dataset[0..max_minibatch_size - 1]
        remaining_dataset = remaining_dataset[max_minibatch_size..-1] || []
        if @minibatch_cache[idx_minibatch].nil?
          @minibatch_cache[idx_minibatch] = {
            # Shape [n_x, minibatch.size]
            x: Numo::DFloat[
              *minibatch.map do |(file, _class_name)|
                Magick::ImageList.new(file).first.export_pixels.map { |color| (color >> 8).to_f / 255.0 }
              end
            ].transpose,
            # Shape [nbr_classes, minibatch.size]
            y: Numo::DFloat[*minibatch.map { |(_file, class_name)| @class_one_hot[class_name] }].transpose
          }
        end
        yield(@minibatch_cache[idx_minibatch][:x], @minibatch_cache[idx_minibatch][:y])
        idx_minibatch += 1
      end
    end

    # Get some classes stats on a given dataset type
    #
    # Parameters::
    # * *dataset_type* (Symbol): Dataset type to loop over (can be :train, :dev or :test)
    # Result:
    # * Hash<String, Hash>: Some statistics, per class name. Here are the available statistics:
    #   * *nbr_files* (Integer): Number of files classified for this class
    def class_stats(dataset_type)
      @classes.to_h do |class_name|
        [
          class_name,
          {
            nbr_files: @datasets[dataset_type].select { |(_file, select_class_name)| select_class_name == class_name }.size
          }
        ]
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
      sample_image = Magick::ImageList.new(@datasets[:train].first[0]).first
      {
        rows: sample_image.rows,
        cols: sample_image.columns,
        channels:
          case sample_image.colorspace
          when Magick::SRGBColorspace
            3
          else
            raise "Unknown colorspace: #{sample_image.colorspace}"
          end
      }
    end

    # Display some statistics in the terminal
    def display_stats
      dataset_types_totals = @types.to_h { |dataset_type| [dataset_type, 0] }
      dataset_types_stats = @types.to_h { |dataset_type| [dataset_type, class_stats(dataset_type)] }
      require 'terminal-table'
      puts(
        Terminal::Table.new(
          title: "Dataset statistics (#{@classes.size} classes)",
          headings: ['Class'] + @types.map(&:to_s) + ['Total']
        ) do |t|
          @classes.each do |class_name|
            dataset_files = @types.map do |dataset_type|
              nbr_files = dataset_types_stats[dataset_type][class_name][:nbr_files]
              dataset_types_totals[dataset_type] += nbr_files
              nbr_files
            end
            t << [class_name] + dataset_files + [dataset_files.sum]
          end
          t.add_separator
          total = dataset_types_totals.values.sum
          t << ['Total'] + @types.map { |dataset_type| dataset_types_totals[dataset_type] } + [total]
          t << [''] + @types.map { |dataset_type| "#{(dataset_types_totals[dataset_type] * 100) / total}%" } + ['100%']
        end
      )
    end

    # Display a sample image
    #
    # Parameters::
    # * *dataset_type* (Symbol): Dataset type from which the sample should be taken
    # * *class_name* (String): Class name from which the sample should be taken
    def display_sample(dataset_type, class_name)
      found_file, found_class_name = @datasets[dataset_type].shuffle.find { |(_file, select_class_name)| select_class_name == class_name }
      puts "Display sample image #{found_file} of class #{found_class_name}"
      display_image(Magick::ImageList.new(found_file).first)
    end

    private

    # Display an image
    #
    # Parameters::
    # * *image* (Magick::Image): The image to be displayed
    def display_image(image)
      require 'tmpdir'
      Dir.mktmpdir do |temp_dir|
        file_name = "#{temp_dir}/display.png"
        image.write(file_name)
        system "start #{file_name}"
        puts 'Press enter to continue...'
        $stdin.gets
      end
    end

  end

end
