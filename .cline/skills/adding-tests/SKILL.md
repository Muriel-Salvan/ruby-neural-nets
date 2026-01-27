---
name: adding-tests
description: Provide rules applicable to any tests modification, creation or deletion, applicable to the test code base (inside the ./spec directory)
---

# adding-tests

ALWAYS start by telling the user "I WILL ALTER TESTS".

1. Add any platform-agnostic test scenarios under the spec/scenarios directory.
2. Add any platform-specific test scenarios under the spec/scenarios.#{RUBY_PLATFORM} directory.
3. Define common test helpers and framework under the spec/ruby_neural_nets_test directory.
4. Group test scenarios per interface kind. Only the following interface kinds are tested:
  * Data loaders are under spec/scenarios/data_loaders.
  * Models are under spec/scenarios/models.
  * Trainers are under spec/scenarios/trainers.
5. Name the test file spec/<scenarios_kind>/<interface_kind>/<class_name>_spec.rb inside its scenarios directory.
6. In the case too many scenarios have to be tested for a single class, then split the scenarios among different files, all part of the same directory, named spec/<scenarios_kind>/<interface_kind>/<class_name>/<scenarios_group_meaning>_spec.rb inside their scenarios directory.
7. Add any shared test scenarios (shared between different spec files) in files named spec/scenarios/<interface_kind>/shared/<common_functionality>_scenarios.rb. Include the shared scenarios in all spec files that have this common functionality, using RSpec's include_examples.
8. ALWAYS respect this structure for a test scenario:
  1. Setup test data and mocks.
  2. Call only the public interface of the class to be tested. Never call private methods of the interface.
  3. Write simple assertions on results of the public interface of the class to be tested. Never assert results from private methods of the interface.
9. Make sure each test scenario run in an isolated way. Running scenarios in whatever order or group should never change the result of the test. If this is not the case, understand what could break the isolation of those unit tests and fix it.

## Usage

You should use this skill anytime you need to add new tests, edit or delete existing tests, and refactor some tests scenarios, helpers or test frameworks.

## Steps

1. Identify which tests need to be added/modified/deleted, for which classes and scenarios.
2. Add/modify/delete test scenarios.
3. Check that all tests are still succeeding, and fix any failure.
4. Identify if there is a need to refactor tests scenarios, helpers or test frameworks. If you see a lot of repeated test code, assertions or scenarios, consider defining test helpers or shared test scenarios to avoid duplication.
5. Check that all tests are still succeeding, and fix any failure.
