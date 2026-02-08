# All common behaviour of the test framework should be part of this file.
# It is automatically required by rspec.

# Needed for FakeFS to work correctly with RSpec and byebug
require 'pp'
require 'byebug/core'

require 'byebug'

require 'ruby_neural_nets_test/helpers'

# Multi-output IO class that writes to multiple destinations
# Used to output logs to both StringIO (for test assertions) and STDOUT (for debugging)
class MultiOutputIO
  def initialize(*ios)
    @ios = ios
  end

  def write(*args)
    @ios.each { |io| io.write(*args) }
  end

  def close
    @ios.each(&:close)
  end
end

RSpec.configure do |config|
  config.include RubyNeuralNetsTest::Helpers
  scenarios_dirs = ['scenarios']
  platform_scenarios_dir = "scenarios.#{RUBY_PLATFORM}"
  scenarios_dirs << platform_scenarios_dir if File.exist?("spec/#{platform_scenarios_dir}")
  config.pattern = scenarios_dirs.map { |dir| "spec/#{dir}/**/*_spec.rb" }.join(',')

  # Set up test logger that captures output to a String instead of STDOUT
  config.before(:each) do
    # Create a StringIO to capture log output for test assertions
    @test_log_stringio = StringIO.new

    # When TEST_DEBUG=1, output to both StringIO and STDOUT and enable debug mode
    # Otherwise, only capture to StringIO
    if ENV['TEST_DEBUG'] == '1'
      # Enable debug mode
      RubyNeuralNets::Logger.debug_mode = true
      # Create a logger that writes to both StringIO (for assertions) and STDOUT (for visibility)
      RubyNeuralNets::Logger.logger = ::Logger.new(MultiOutputIO.new(@test_log_stringio, STDOUT))
    else
      # Default behavior: only capture to StringIO
      RubyNeuralNets::Logger.logger = ::Logger.new(@test_log_stringio)
    end
  end
end
