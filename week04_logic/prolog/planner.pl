% CS F407 Lab, Week 4 | Author: Samar Talwar | Not licensed for reuse or submission by others.
% Prolog knowledge base for warehouse planning verification.

% --- Task 6: Warehouse connectivity ---
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

can_move(X, Y) :- connected(X, Y).

% --- Task 7: Valid move rule ---
valid_move(X, Y) :- connected(X, Y).

% --- Task 8: Logical reasoning chain ---
wet_road.
slippery :- wet_road.
reduce_speed :- slippery.

% --- Task 8: Penguin/bird/animal ---
penguin(polly).
bird(X) :- penguin(X).
animal(X) :- bird(X).
