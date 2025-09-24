require 'ruby-prof'
require 'fileutils'

module RubyNeuralNets

  class Profiler

    # Constructor
    #
    # Parameters::
    # * *profile* (Boolean): Should we profile the execution of each epoch? [default: false]
    def initialize(profile: false)
      @profile = profile
    end

    # Profile an epoch
    #
    # Parameters::
    # * *idx_epoch* (Integer): Epoch index that we are profiling
    # * *code_to_be_profiled* (Block): Code to be called to train for this epoch
    def profile(idx_epoch, &code_to_be_profiled)
      (
        if @profile
          proc do |&code|
            profiling_result = RubyProf::Profile.profile { code.call }
            RubyProf::FlatPrinter.new(profiling_result).print(STDOUT)
            printer = RubyProf::CallStackPrinter.new(profiling_result)
            FileUtils.mkdir_p 'profiling'
            File.open("profiling/epoch_#{idx_epoch}.html", 'w') { |f| printer.print(f, {}) }
          end
        else
          proc { |&code| code.call }
        end
      ).call(&code_to_be_profiled)
    end

  end

end
