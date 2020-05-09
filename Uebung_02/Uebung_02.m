%% Aufgabe a
syms x
% definiton function
func_1 = -4*x*exp(-2*x^2)+3*cos(x) == -3;
func_2 = -4*x*exp(-2*x^2)+3*cos(x) == -5;
% solve
x_1 = vpasolve(func_1, x);
x_2 = vpasolve(func_2, x);

% plot
x_num = linspace(-4, 2);
y = -4*x_num.*exp(-2*x_num.^2)+3.*cos(x_num)+3;
y_lim_h = linspace(0, 0);
y_lim_l = linspace(-2, -2);

figure()
plot(x_num, y, x_num, y_lim_h, x_num, y_lim_l)

%% Aufgabe b
func_2 = -1./(3-4*x*exp(-2*x^2)+3*cos(x)) == 0;

x_num = -0.1588023735798893;

% compute a and roundoff in 8 Dezimalstellen
relation_a = -1/(-4*x_num.*exp(-2*x_num.^2)+3.*cos(x_num)+3);
relation_a = vpa(relation_a, 8) 

% compute y and roundoff in 8 Dezimalstellen
y = -4*x_num.*exp(-2*x_num.^2)+3.*cos(x_num)+3;
y = vpa(y, 8) 

% final result 
g_b = abs(relation_a*y+1)


%% Aufgabe d
% initialize
x_a = -1.5;
% Newton Methode
loesung_c_newton = zeros(4,2);
for i = 1:4
    x_b = x_a - (3*x_a + exp(-2*x_a^2) + 3*sin(x_a))/(3 + (-4*x_a)*exp(-2*x_a^2) +3*cos(x_a));
    loesung_c_newton(i,1) = x_a;
    loesung_c_newton(i,2) = x_b;
    x_a = x_b;
end

% Fixpunkt Methode
x_a = -1.5;
loesung_c_fix = zeros(4,2);
relation_a = -1/(-4*x_num.*exp(-2*x_num.^2)+3.*cos(x_num)+3);
for i = 1:4
    x_b = x_a + relation_a*(3*x_a + exp(-2*x_a^2) + 3*sin(x_a));
    loesung_c_fix(i,1) = x_a;
    loesung_c_fix(i,2) = x_b;
    x_a = x_b;
end

%% Aufgabe f
% initialize
x_a = -3;
% Newton Methode
loesung_c_newton = zeros(4,2);
for i = 1:50
    x_b = x_a - (3*x_a + exp(-2*x_a^2) + 3*sin(x_a))/(3 + (-4*x_a)*exp(-2*x_a^2) +3*cos(x_a));
    loesung_c_newton(i,1) = x_a;
    loesung_c_newton(i,2) = x_b;
    x_a = x_b;
end

% Fixpunkt Methode
x_a = -3;
loesung_c_fix = zeros(4,2);
relation_a = -1/(-4*x_num.*exp(-2*x_num.^2)+3.*cos(x_num)+3);
for i = 1:50
    x_b = x_a + relation_a*(3*x_a + exp(-2*x_a^2) + 3*sin(x_a));
    loesung_c_fix(i,1) = x_a;
    loesung_c_fix(i,2) = x_b;
    x_a = x_b;
end

loesung_c_vergleich = loesung_c_fix;
loesung_c_vergleich(:,1) = loesung_c_newton(:,2);