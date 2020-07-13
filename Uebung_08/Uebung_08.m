%% Aufgabe 2
m = 1;
Mess_1 = [0, 1, 2, 3, 4, 5;
                    0, 0.31, 0.78, 2.17, 4.12, 5.86];
d_phi_1= sum(Mess_1(1, :).^4/(4*m));
d_phi_2= sum(Mess_1(1, :).^2/(2*m) .*Mess_1(2, :));
f = d_phi_2/d_phi_1;
x_heat = f/(2*m)*(Mess_1(1,:).^2);

Mess_2 = [0, 0.305, 0.71, 2.12, 4.18, 5.84];

delta = x_heat - Mess_2;
sum(delta.^2)/length(delta);

f/(2*m)*100^2