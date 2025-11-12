# All common behaviour of the test framework should be part of this file.
# It is automatically required by rspec.

# Needed for FakeFS to work correctly with RSpec
require 'pp'
require 'byebug'

require 'ruby_neural_nets_test/helpers'

RSpec.configure do |config|
  config.include RubyNeuralNetsTest::Helpers
end
