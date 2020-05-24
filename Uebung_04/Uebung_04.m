%% Uebung_01 d)
l = 2.5;
J = 0.05;
m = 1.5;
g = 9.81;
c = 0.01;

A = [0, 1; g/(J+m*l^2), -c/(J+m*l^2)];

[eig_val, eig_vec] = eig(A)