# Test rules, applicable to the test code base (inside the ./spec directory)

1. Platform-agnostic unit test scenarios are all defined under the spec/scenarios directory.
2. Platform-specific unit test scenarios are all defined under the spec/scenarios.#{RUBY_PLATFORM} directory.
3. Unit test framework and helpers are all defined under the spec/ruby_neural_nets_test directory.
4. Unit test scenarios are grouped per kind of interface being tested. Only the following kinds are tested:
  * Under spec/scenarios/data_loaders: All unit tests testing data loaders.
  * Under spec/scenarios/models: All unit tests testing models.
  * Under spec/scenarios/trainers: All unit tests testing trainer.
5. Unit test spec file name should follow the template spec/scenarios/<interface_kind>/<class_name>_spec.rb .
6. In the case too many scenarios have to be tested for a single class, then the scenarios will be split among different files all part of the same directory, following this template: spec/scenarios/<interface_kind>/<class_name>/<scenarios_group_meaning>_spec.rb .
7. Common unit test scenarios (shared between different spec files) should be defined in new files named spec/scenarios/<interface_kind>/shared/<common_functionality>_scenarios.rb and will be included in all spec files that have this common functionality.
8. Each unit test scenario should have the same structure:
  1. Setup test data and mocks.
  2. Call only the public interface of the class to be tested. Never call private methods of the interface.
  3. Write simple assertions on results of the public interface of the class to be tested. Never assert results from private methods of the interface.
9. Each unit test scenario should run in an isolated way: running scenarios in whatever order or group should never change the result of the test. If side effects are found while running test scenarios that are impacting other test scenarios, those side effects should be removed to make sure each test scenario runs in isolation.
10. RSpec tests can be run in WSL using the command line `.\tools\rspec_wsl.cmd`. It accepts the same CLI arguments as rspec itself. For example, a single test scenario can be run using `.\rspec_wsl.cmd -e "tracks progress and reports correct cost and accuracy"`.
11. Successful tests run should output "0 failures" at the end. If this string is not found then it means tests are failing.
