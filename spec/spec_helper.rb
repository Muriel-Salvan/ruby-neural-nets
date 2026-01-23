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
    stringio = StringIO.new
    # Create a new logger that writes to the StringIO
    test_logger = ::Logger.new(stringio)
    # Store the StringIO in thread-local storage for access by helpers
    Thread.current[:test_log_stringio] = stringio
    # Replace the global logger with our test logger
    RubyNeuralNets::Logger.logger = test_logger
  end

  # Clean up after each test
  config.after(:each) do
    # Clear the thread-local storage
    Thread.current[:test_log_stringio] = nil
    # Reset to default STDOUT logger (optional, but good practice)
    RubyNeuralNets::Logger.logger = ::Logger.new(STDOUT)
  end
end