
rusting(body, X) :- X = 1.
rusting(pipes, X) :- X = 1.
rusting(joints, X) :- X = 1.

less_thickness(body, X) :- X < 10.
less_thickness(pipes, X) :- X < 8.
less_thickness(joints, X) :- X < 6.


parts(body).
parts(pipes).
parts(joints).
