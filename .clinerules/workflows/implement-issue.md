<task name="Implement issue">

<task_objective>
Implement all the requirements defined in a given Github issue, while following all the rules of this project.
</task_objective>

<detailed_sequence_steps>
# Implement issue Process - Detailed Sequence of Steps

## 1. Get issue requirements

1. Use the `ask_followup_question` command to ask the USER which Github issue should be implemented.

2. Use the `bundle exec ruby ./tools/github/issue_details {{issue_number}}` CLI to retrieve all the details of this Github issue.

## 2. Come up with an implementation plan

1. Analyze the current code structure and content to understand how the Github issue should be implemented.

2. Analyze all the rules that you should adhere to when implementing a task.

3. Inform the USER of your implementation plan, taking into consideration the issue requirements, the current code, all the rules to be respected and possible USER inputs.

4. Use the `ask_followup_question` command to ask the USER if this plan is valid.

5. If the USER agrees with this plan, go on to section 3. Otherwise correct your implementation plan by repeating the steps of this section 2.

## 3. Implement the issue following the implementation plan

1. Perform all the agreed steps in the implementation plan to implement the issue.

2. ALWAYS make sure that all the coding and development rules and steps are respected. If rules require you to run tests, commit, push changes... don't forget about them. If you are not sure, double-check all the rules that are defined.

3. Perform a final verification of the implementation plan against all the actions you did. If you think some steps of the implementation plan were not implemented properly or are missing, fix it by repeating the steps of this section 3.

</detailed_sequence_steps>

</task>
