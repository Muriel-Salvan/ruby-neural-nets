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
end
