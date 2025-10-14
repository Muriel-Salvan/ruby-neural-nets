module RubyNeuralNets
  
  class Options

    # Constructor
    def initialize
      # Require all classes that can be selectable in the options
      %w[
        models
        data_loaders
        accuracies
        losses
        optimizers
      ].each do |component_type|
        Dir.glob("#{__dir__}/#{component_type}/*.rb") do |file|
          # Check that this file is not platform specific
          base_name = File.basename(file, '.rb')
          _base_base_name, arch = base_name.split('.')
          require "ruby_neural_nets/#{component_type}/#{base_name}" if arch.nil? || arch == RUBY_PLATFORM
        end
      end
      # Description of options, with their default values, per option name.
      # Options can be changed by CLI arguments.
      # The description of an option is either its default value, or a Hash with more information:
      # * *value* (Object): The default value of the option. Is the default property used when not specifying a Hash for the option.
      # * *from* (Module): In the case values are classes, give the module containing the possible classes for the value.
      # * *ancestor* (Class): In the case values are classes, give the ancestor possible classes for the value should have. Mandatory if :from is also specified.
      # * *name* (String): In the case values are classes, give the default class name (when specifying name, value is not needed).
      # * *desc* (String or Array<String>): Additional description of the option.
      # * *options* (Hash): Sub-options that are linked to this option. The structure is the same as the options object itself.
      @options = {
        accuracy: {
          from: RubyNeuralNets::Accuracies,
          ancestor: RubyNeuralNets::Accuracy,
          name: 'ClassesNumo'
        },
        data_loader: {
          from: RubyNeuralNets::DataLoaders,
          ancestor: RubyNeuralNets::DataLoader,
          name: 'Numo',
          options: {
            max_minibatch_size: 5000,
            dataset: {
              desc: "Possible values are #{Dir.glob('./datasets/*').map { |file| File.basename(file) }.join(', ')}.",
              value: 'colors'
            }
          }
        },
        loss: {
          from: RubyNeuralNets::Losses,
          ancestor: RubyNeuralNets::Loss,
          name: 'CrossEntropy'
        },
        model: {
          from: RubyNeuralNets::Models,
          ancestor: RubyNeuralNets::Model,
          name: 'OneLayer',
          options: {
            layers: [100]
          }
        },
        optimizer: {
          from: RubyNeuralNets::Optimizers,
          ancestor: RubyNeuralNets::Optimizer,
          name: 'Adam',
          options: {
            decay: 0.9,
            learning_rate: 0.001
          }
        },
        nbr_epochs: 100,
        instability_checks: {
          desc: [
            'Possible values are:',
            '* byebug: Open a byebug prompt for debugging.',
            '* exception: Raise an exception.',
            '* off: Turn off checks.',
            '* warning: Display a warning.'
          ],
          value: :byebug
        },
        gradient_checks: {
          desc: [
            'Possible values are:',
            '* byebug: Open a byebug prompt for debugging.',
            '* exception: Raise an exception.',
            '* off: Turn off checks.',
            '* warning: Display a warning.'
          ],
          value: :exception
        },
        profiling: false
      }.to_h { |option, option_info| [option, normalize_option_info(option_info)] }
    end

    # Parse CLI arguments for options
    def parse_cli
      # Parse command-line options
      require 'optparse'
      OptionParser.new do |opts|
        opts.banner = "Usage: run [options]"
        opts.separator ''
        opts.separator 'Global options:'
        opts.on(
          '-h', '--help',
          'Display usage and exit'
        ) do
          puts opts
          exit 0
        end
        # Group all non classes options in global options
        @options.select { |_option, option_info| !option_info.key?(:known_classes) }.each do |option, option_info|
          add_option(opts, option, option_info)
        end
        @options.select { |_option, option_info| option_info.key?(:known_classes) }.each do |option, option_info|
          opts.separator ''
          opts.separator "Options for #{option}:"
          add_option(opts, option, option_info)
        end
      end.parse!
    end

    # Return the value of an option
    #
    # Parameters::
    # * *options* (Symbol or Array<Symbol>): The option to retrieve. Can be nested in the case of sub-options.
    # Result::
    # * Object: The corresponding value
    def [](*options)
      info(*options)[:value]
    end

    # Create a new instance of a class defined in the options, passing arguments and sub-options properly to the constructor.
    #
    # Parameters::
    # * *options* (Symbol or Array<Symbol>): The option to retrieve. Can be nested in the case of sub-options.
    # * *args* (Array): List of arguments to give the constructor.
    # * *kwargs* (Array): List of kwargs to give the constructor. They will be completed with default ones from sub-options if any.
    # Result::
    # * Object: The corresponding instance
    def instantiate(options, *args, **kwargs)
      option_info = info(*options)
      option_info[:value].new(
        *args,
        **option_info[:known_classes][option_info[:value]].to_h { |kwarg| [kwarg, self[*options, kwarg]] }.merge(kwargs)
      )
    end

    private

    # Return an option info, selected from a list of option names (nested in case of sub-options).
    # For example: info(:model, :layers) will return @options[:model][:options][:layers]
    #
    # Parameters::
    # * *options* (Symbol or Array<Symbol>): The option to retrieve. Can be nested in the case of sub-options.
    # Result::
    # * Object: The corresponding value
    def info(*options)
      current_options = nil
      next_options = @options
      [options].flatten(1).each do |option|
        current_options = next_options[option]
        next_options = next_options[option][:options]
      end
      current_options
    end
        
    # Add an option to the options parser
    #
    # Parameters::
    # * *opts* (OptionsParser): The options parser to complete
    # * *option* (Symbol): The option to add
    # * *option_info* (Hash): The corresponding option information
    def add_option(opts, option, option_info)
      format = option_format(option_info)
      opts.on(
        *(
          [
            "--#{option.to_s.gsub('_', '-')} #{format.upcase.gsub(' ', '_')}",
            "Specify the #{option} to use."
          ] +
            option_info[:desc] +
            [
              "Format is #{format}.",
              "Defaults to #{option_str(option_info[:value], option_info)}."
            ]
        )
      ) { |value| option_info[:value] = parse_option_str(value, option_info) }
      option_info[:options].each do |sub_option, sub_option_info|
        add_option(opts, sub_option, sub_option_info)
      end
    end
    
    # Get a map of class names that belong to a module and inherit from an ancestor
    #
    # Parameters::
    # * *mod* (Module): The module from which we look for classes
    # * *ancestor* (Class): The ancestor classes should belong to
    # Result::
    # * Hash<Class, Array<Symbol>>: For each class, the list of option kwargs that can be used with the constructor
    def discover_classes_of(mod, ancestor)
      mod.constants.map do |const|
        model_constant = mod.const_get(const)
        if model_constant.is_a?(Class) && model_constant.ancestors.include?(ancestor)
          [
            model_constant,
            model_constant.
              instance_method(:initialize).
              parameters.
              select { |(arg_type, _arg_name)| arg_type == :keyreq }.
              map { |(_arg_type, arg_name)| arg_name }
          ]
        else
          nil
        end
      end.compact.to_h
    end

    # Get a string representation of an option
    #
    # Parameters::
    # * *value* (Object): The option value to represent
    # * *option_info* (Hash): The option info
    # Result::
    # * String: Corresponding string
    def option_str(value, option_info)
      if option_info.key?(:from)
        value.name.split('::').last
      elsif value.is_a?(Array)
        value.join(',')
      else
        value.to_s
      end
    end

    # Set an option from one of its string representations
    #
    # Parameters::
    # * *str* (String): The option string representation
    # * *option_info* (Hash): The option info
    # Result::
    # * Object: Corresponding option value
    def parse_option_str(str, option_info)
      if option_info.key?(:from)
        option_info[:from].const_get(str.to_sym)
      else
        value = option_info[:value]
        if value.is_a?(Array)
          elem_option_info = { value: value.first }
          str.split(',').map { |elem_str| parse_option_str(elem_str, elem_option_info) }
        elsif value.is_a?(Integer)
          Integer(str)
        elsif value.is_a?(Float)
          str.to_f
        elsif value.is_a?(Symbol)
          str.to_sym
        elsif value.is_a?(String)
          str
        elsif value.is_a?(TrueClass) || value.is_a?(FalseClass)
          str == 'true'
        else
          raise "Can't parse value \"#{str}\" of type #{value.class}"
        end
      end
    end

    # Get a descriptive option format
    #
    # Parameters::
    # * *option_info* (Hash): The option info
    # Result::
    # * String: Corresponding description of its format
    def option_format(option_info)
      if option_info.key?(:from)
        "#{option_info[:from].name.split('::').last} name"
      else
        value = option_info[:value]
        if value.is_a?(Array)
          "comma-separated list of #{option_format({ value: value.first })}"
        elsif value.is_a?(Integer)
          'integer'
        elsif value.is_a?(Float)
          'float'
        elsif value.is_a?(Symbol) || value.is_a?(String)
          'string'
        elsif value.is_a?(TrueClass) || value.is_a?(FalseClass)
          'boolean'
        else
          raise "Can't parse option value \"#{value}\" of type #{value.class}"
        end
      end
    end

    # Normalize an options information.
    # Make sure all mandatory properties are set, using the Hash form.
    #
    # Parameters::
    # * *option_info* (Object): An option info (as described from the options object).
    # * *additional_desc* (Array): Additional description to be added to those options [default: []]
    # Result::
    # * Hash<Symbol, Object>: The corresponding normalized option information
    def normalize_option_info(option_info, additional_desc: [])
      # Normalize the configuration
      normalized_info = option_info.is_a?(Hash) ? option_info : { value: option_info }
      normalized_info[:desc] = [] unless normalized_info.key?(:desc)
      normalized_info[:desc] = [normalized_info[:desc]] unless normalized_info[:desc].is_a?(Array)
      if normalized_info.key?(:from)
        # In the case values are classes, give the set of possible class names and their corresponding needed options at construction time.
        # Hash< String, Array<Symbol> >
        normalized_info[:known_classes] = discover_classes_of(normalized_info[:from], normalized_info[:ancestor])
        normalized_info[:desc] << "Possible values are #{classes_desc(normalized_info[:known_classes].keys)}."
      end
      normalized_info[:desc].concat(additional_desc)
      # Fill value from name if needed
      normalized_info[:value] = normalized_info[:from].const_get(normalized_info[:name].to_sym) if normalized_info.key?(:name)
      normalized_info[:options] = (normalized_info[:options] || {}).to_h do |sub_option, sub_option_info|
        [
          sub_option,
          normalize_option_info(
            sub_option_info,
            additional_desc: additional_desc +
              (option_info.key?(:known_classes) ? ["This is used by #{classes_desc(option_info[:known_classes].keys.select { |class_name| option_info[:known_classes][class_name].include?(sub_option) })}."] : [])
          )
        ]
      end
      normalized_info
    end

    # Get a description of a list of classes
    #
    # Parameters::
    # * *classes* (Array<Class>): List of classes to represent
    # Result::
    # * String: Corresponding representation
    def classes_desc(classes)
      classes.map { |c| c.name.split('::').last }.join(', ')
    end

  end

end