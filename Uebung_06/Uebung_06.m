%% Aufgabe 1 c
s = dsolve('D2y = -2*Dy -5*y', 'Dy(0) = 1', 'y(0) = 0')

%% Aufgabe 1 d
t = linspace(0, 10, 1e6);
y = (sin(2*t).*exp(-t))/2;
plot(t, y, t, zeros(size(t)))
hold on
scatter(t(y==max(y)), max(y), 'r*')
scatter(t(y==min(y)), min(y), 'r*')
grid on
xlabel('Time in s')
ylabel('Phi in rad')
title('Schwingungsantwort')
legend('Antwort','Stabil', 'Phi Max', 'Phi Min')

t_max = t(y==max(y))
t_min = t(y==min(y))
y_max = max(y) - min(y)
