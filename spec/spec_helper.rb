# All common behaviour of the test framework should be part of this file.
# It is automatically required by rspec.

# Needed for FakeFS to work correctly with RSpec and byebug
require 'pp'
require 'byebug/core'

require 'byebug'

require 'ruby_neural_nets_test/helpers'

RSpec.configure do |config|
  config.include RubyNeuralNetsTest::Helpers
  scenarios_dirs = ['scenarios']
  platform_scenarios_dir = "scenarios.#{RUBY_PLATFORM}"
  scenarios_dirs << platform_scenarios_dir if File.exist?("spec/#{platform_scenarios_dir}")
  config.pattern = scenarios_dirs.map { |dir| "spec/#{dir}/**/*_spec.rb" }.join(',')

  # Set up test logger that captures output to a String instead of STDOUT
  config.before(:each) do
    # Create a StringIO to capture log output
    @test_log_stringio = StringIO.new

    if ENV['TEST_DEBUG'] == '1'
      # Enable debug mode when TEST_DEBUG is set
      RubyNeuralNets::Logger.debug_mode = true
      # Create a logger that writes to both StringIO (for assertions) and STDOUT (for debugging)
      RubyNeuralNets::Logger.logger = ::Logger.new(@test_log_stringio).tap do |logger|
        # Add a second log device that writes to STDOUT
        original_add = logger.method(:add)
        logger.define_singleton_method(:add) do |severity, message = nil, progname = nil, &block|
          result = original_add.call(severity, message, progname, &block)
          # Also output to STDOUT for debugging
          formatted_message = if message.nil?
                                block ? block.call : progname
                              else
                                message
                              end
          puts "[#{severity}] #{formatted_message}" if formatted_message
          result
        end
      end
    else
      # Normal test mode: only capture to StringIO
      RubyNeuralNets::Logger.logger = ::Logger.new(@test_log_stringio)
    end
  end
end
