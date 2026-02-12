# All common behaviour of the test framework should be part of this file.
# It is automatically required by rspec.

# Needed for FakeFS to work correctly with RSpec and byebug
require 'pp'
require 'byebug/core'

require 'byebug'

require 'ruby_neural_nets_test/helpers'

# IO wrapper that writes to multiple destinations
class MultiIO
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
    # Create a StringIO to capture log output
    @test_log_stringio = StringIO.new
    # Check if TEST_DEBUG is enabled
    test_debug = ENV['TEST_DEBUG'] == '1'
    # Enable debug mode if TEST_DEBUG is set
    RubyNeuralNets::Logger.debug_mode = test_debug
    # Replace the global logger with our test logger
    # When TEST_DEBUG=1, also output to STDOUT for visibility
    logger_io = test_debug ? MultiIO.new(@test_log_stringio, STDOUT) : @test_log_stringio
    RubyNeuralNets::Logger.logger = ::Logger.new(logger_io)
  end
end
