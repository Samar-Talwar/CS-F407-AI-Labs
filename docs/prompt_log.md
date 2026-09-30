# CS F407 Lab, Week 4 | Author: Samar Talwar | Not licensed for reuse or submission by others.
## Prompt Log Entry

**Date**: 2026-09-30  
**Week**: 04  
**Prompt**: I want to implement a simple planning agent in Python.
Represent a state as a set of logical propositions.
Each action should contain:
• a name;
• positive preconditions;
• negative preconditions;
• positive effects;
• negative effects.
An action is applicable if all of its preconditions are satisfied by the current state.
When an action is applied:
1. remove its negative effects from the state;
2. add its positive effects;
Use breadth-first search to find a sequence of actions that achieves a specified goal.
The program should also:
• detect when no plan exists;
• print the resulting sequence of actions;
• print the states reached after each action.
Explain the implementation and identify any assumptions you make.
Run the generated program on the warehouse problem.

**What was changed after review**:
- Added `missing_preconditions()` helper for Task 0 diagnostics
- Added `neg_pre` and `neg_eff` fields to Action to fully support negative preconditions and effects
- Made `applicable()` and `apply_action()` exactly match the spec (remove neg_eff then add pos_eff)
- Added independent validator that re-implements applicable/apply logic inline (no code sharing)
- Added BFS planner that returns plan, states after each action, and nodes expanded
- Added grounding of warehouse actions (Move between A-B, B-A, B-C, C-B; PickUp/Drop at A,B,C)
- Added support for negative preconditions/effects via synthetic problem (UnlockDoor/Enter)
- Added Prolog integration via subprocess calls to swipl for independent verification
- Added cross-check: every Move step in Python plan must be accepted by Prolog valid_move/2
- Added proper handling of swipl absence (visible skip reason, not silent pass)