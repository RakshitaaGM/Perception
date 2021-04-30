%(2)To find F_low and f_high (half power frequencies)
A = ones(1,500);
B = zeros(1,500);
L = 1000;
C = [A B]
Fs = 1000000;
[bw,f_low,f_high, power] = powerbw(C, Fs);% Sampling frequency                    
disp(f_low)
disp(f_high)

