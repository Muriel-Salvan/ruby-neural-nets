# General rules, applicable to the whole code base

1. Every class should be defined in a single file that follows the module and class name in its path, using snake_case for files and paths.
2. Classes always define their public methods first, followed by their private ones below. The keyword private is separating both parts.
3. Text files should always end with a carriage return character.
4. Avoid defining local variables that are used only once. Try to use the variable value directly where it is needed (chained calls are preferred to explicit variables definitions).
5. Avoid catching cases of missing or unknown data explicitely: if the data is not in the expected format then a normal exception should be raised, without having to add extra code to support it. For example when accessing a hash's value that is supposed to exist, don't test for its presence (no need for "next if hash[key].nil?").
6. Any code or test modification should be validated by running all the tests (not only the modified ones). It is expected that it runs without any error, with the mention "0 failures" at the end of the output.
7. Any code or test modification should trigger a verification of the README.md file content, and make sure that all sections still have up-to-date content regarding the change.
8. Module names should not be added to class names, constants and class method calls when used inside the module itself. For example, no need to add MyModule:: in front of any class that is used inside the MyModule module.
