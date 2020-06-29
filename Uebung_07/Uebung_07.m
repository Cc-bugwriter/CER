%% Aufgabe 1a
A = [-4 -2; 1 -2];
lambda = eig(A);

A = [-400 0; 1 -2];
lambda = eig(A);
%% Aufgabe 1c
t = 0;
% A =  [-4 -2 0; 1 -2 0; 0 0 0];
A =  [-400 0 0; 1 -2 0; 0 0 0];
% B = [4*sin(t); sin(t)-2*cos(t); 1];
B = [400*sin(t); sin(t)-2*cos(t); 1];
x_0 = [0.1; 5; 0];

for i = 1:3
    x = x_0;
    t = x(3);
    x_0 = x+ 0.1.*(A*x + B)
    X(i,:) = x_0;
end

%% Aufgabe 1d
t = 0;
% A =  [-4 -2 0; 1 -2 0; 0 0 0];
A =  [-400 0 0; 1 -2 0; 0 0 0];
% B = [4*sin(t); sin(t)-2*cos(t); 1];
B = [400*sin(t); sin(t)-2*cos(t); 1];
x_0 = [0.1; 5; 0];

for i = 1:300
    x = x_0;
    t = x(3);
    x_0 = x+ 0.001.*(A*x + B);
    if x_0(3) == 100 || x_0(3) == 200 || x_0(3) == 300
        pri
    end
    X(i,:) = x_0;
end

%% Aufgabe 1e
x_0 = [0.1; 5; 0];

h = 0.1;
t = x_0(3);

dF = [-400*h-1 0 400*h*cos(t); h -2*h-1 h*cos(t)+2*h*sin(t); 0 0 -1]
A = [-400*h-1 0 0; h -2*h-1 0; 0 0 -1];
B = h*[400*sin(t); sin(t)-2*cos(t); 1];

F = x_0 + A*x_0 + B;

x_neu = x_0 - inv(dF)*F

%% Aufgabe 2a
h = 4e-5;
y_heat_0 = [0; 0];
for i = 1:1.125e5
y_1 = y_heat_0(1);
y_2 = y_heat_0(2);

f = [1; sin(4/3*y_1 +2) + 1/4*y_2 + 2/5];
y_heat_1 = y_heat_0 + h*f;

y_true(i,:) = y_heat_1';

y_heat_0  = y_heat_1;
end
plot(y_true(:,1), y_true(:,2))
xlim([0 4.5])
ylim([-3 3])
grid on
hold on
y = [];


h = 2;
y_heat_0 = [0; 0];
y(1,:) = y_heat_0';
for i = 2:3
y_1 = y_heat_0(1);
y_2 = y_heat_0(2);

f = [1; sin(4/3*y_1 +2) + 1/4*y_2 + 2/5];
y_heat_1 = y_heat_0 + h*f;

y(i,:) = y_heat_1';

y_heat_0  = y_heat_1;
end
plot(y(:,1), y(:,2), 'r')
xlim([0 4.5])
ylim([-3 3])
title("Geometrische Konstuktion explizites Euler-Verfahren")

%% Aufgabe 2b
y_heat_0 = [0; 0];
y(1,:) = y_heat_0';
for i = 2:3
y_1 = y_heat_0(1);
y_2 = y_heat_0(2);

f = [1; sin(4/3*y_1 +2) + 1/4*y_2 + 2/5];

y_heat_1 = y_heat_0 + h*f;

y_1 = y_heat_1(1);
f = [1; sin(4/3*y_1 +2) + 2/5];

y_heat_1 = (y_heat_0 + h*f)*2;
y_2 = y_heat_1(2);

y(i,:) = [y_1, y_2];

y_heat_0 = y(i,:)';
end
plot(y_true(:,1), y_true(:,2))
grid on
hold on
title("Geometrische Konstuktion implizites Euler-Verfahren")
plot(y(:,1), y(:,2), 'r')
xlim([0 4.5])
ylim([-3 3])
y = [];

%% Aufgabe 2c
y_heat_0 = [0; 0];
y(1,:) = y_heat_0';
for i = 2:3
y_1 = y_heat_0(1);
y_2 = y_heat_0(2);

s_1 = [1; sin(4/3*y_1 +2) + 1/4*y_2 + 2/5];
s_2 = [1; sin(4/3*(y_1+h*s_1(1)) +2) + 1/4*(y_2+h*s_1(2)) + 2/5];

y_heat_1 = y_heat_0 + h/2*(s_1 + s_2);

y(i,:) = y_heat_1';
y_heat_0  = y_heat_1;
end
plot(y_true(:,1), y_true(:,2))
grid on
hold on
title("Geometrische Konstuktion Heun-Verfahren")
plot(y(:,1), y(:,2), 'r')
xlim([0 4.5])
ylim([0 3])
y = [];

%% Aufgabe 2d
y_heat_0 = [0; 0];
y(1,:) = y_heat_0';
for i = 2:3
y_1 = y_heat_0(1);
y_2 = y_heat_0(2);

s_1 = [1; sin(4/3*y_1 +2) + 1/4*y_2 + 2/5];
s_2 = [1; sin(4/3*(y_1+h/2*s_1(1)) +2) + 1/4*(y_2+h/2*s_1(2)) + 2/5];
s_3 = [1; sin(4/3*(y_1+h/2*s_2(1)) +2) + 1/4*(y_2+h/2*s_2(2)) + 2/5];
s_4 = [1; sin(4/3*(y_1+h*s_3(1)) +2) + 1/4*(y_2+h*s_3(2)) + 2/5];

y_heat_1 = y_heat_0 + h/6*(s_1 + 2*s_2 + 2*s_3 + s_4);

y(i,:) = y_heat_1';
y_heat_0  = y_heat_1;
end
plot(y_true(:,1), y_true(:,2))
grid on
hold on
title("Geometrische Konstuktion Runge-Kutta-Verfahren")
plot(y(:,1), y(:,2), 'r')
xlim([0 4.5])
ylim([0 3])
y = [];