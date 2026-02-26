# All common behaviour of the test framework should be part of this file.
# It is automatically required by rspec.

# Needed for FakeFS to work correctly with RSpec and byebug
require 'pp'
require 'byebug/core'

require 'byebug'

require 'ruby_neural_nets_test/helpers'

# IO adapter that writes to multiple targets simultaneously.
# Used to tee log output to both StringIO (for test assertions) and STDOUT (for debugging).
class TeeIO

  # Create a new TeeIO that delegates writes to all given targets
  #
  # Parameters::
  # * *targets* (Array<IO>): IO objects to write to
  def initialize(*targets)
    @targets = targets
  end

  # Write data to all targets
  #
  # Parameters::
  # * *args* (Array): Arguments to pass to each target's write method
  # Result::
  # * Integer: Number of bytes written (from the first target)
  def write(*args)
    @targets.each { |t| t.write(*args) }
  end

  # Close all targets
  def close
    @targets.each(&:close)
  end

end

RSpec.configure do |config|
  config.include RubyNeuralNetsTest::Helpers
  scenarios_dirs = ['scenarios']
  platform_scenarios_dir = "scenarios.#{RUBY_PLATFORM}"
  scenarios_dirs << platform_scenarios_dir if File.exist?("spec/#{platform_scenarios_dir}")
  config.pattern = scenarios_dirs.map { |dir| "spec/#{dir}/**/*_spec.rb" }.join(',')

  # Set up test logger that captures output to a String instead of STDOUT.
  # When TEST_DEBUG=1, also enable debug mode and tee output to STDOUT.
  config.before(:each) do
    @test_log_stringio = StringIO.new
    if ENV['TEST_DEBUG'] == '1'
      RubyNeuralNets::Logger.debug_mode = true
      RubyNeuralNets::Logger.logger = ::Logger.new(TeeIO.new(@test_log_stringio, STDOUT))
    else
      RubyNeuralNets::Logger.debug_mode = false
      RubyNeuralNets::Logger.logger = ::Logger.new(@test_log_stringio)
    end
  end
end
