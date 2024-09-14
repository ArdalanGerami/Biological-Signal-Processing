%% part a
length = 35;
n = 1:length;
x(n) = 0.8.^n.*heaviside(n)-0.5.*0.8.^(n-1).*heaviside(n-1);
fft_x = fft(x);
complex_cepstrum_recursive = [];
complex_cepstrum_definition = ifft(log(fft_x));
complex_cepstrum = cceps(x);
complex_cepstrum_recursive(1) = log(x(1));
for i=2:length
    a = 0;
    for j=1:i-1
        a = a + (j-1)/(i-1)*complex_cepstrum_recursive(j)*x(i-(j-1))/x(1);
    end
   complex_cepstrum_recursive(i) = x(i)/x(1) - a; 
end
figure;
subplot(3,1,1)
plot(complex_cepstrum)
title('complex cepstrum by matlab cceps')
ylabel('x\^[n]')
xlabel('n')
subplot(3,1,2)
plot(complex_cepstrum_definition )
title('complex cepstrum by DFT definition')
ylabel('x\^[n]')
xlabel('n')
subplot(3,1,3)
plot(complex_cepstrum_recursive)
title('complex cepstrum by recursive definition')
ylabel('x\^[n]')
xlabel('n')
%% part b
% disease : cardiac 
ecg_p1 = readtable("207-try.csv");
% disease : Myocardial infarction
ecg_p2 = load('s0001.mat');
% healthy control
ecg_h = load('s0285.mat');
figure;
subplot(3,1,1)
plot(ecg_p1.heart(640:1400))
title('ECG for patient (cardiac)')
ylabel('x[n]')
xlabel('n')
subplot(3,1,2)
plot(cceps(ecg_p1.heart(640:1400)))
title('complex cepstrum for patient (cardiac)')
ylabel('x^[n]')
xlabel('n')
subplot(3,1,3)
plot(rceps(ecg_p1.heart(640:1400)))
title('real cepstrum for patient (cardiac)')
ylabel('c_x[n]')
xlabel('n')
figure;
subplot(3,1,1)
plot(ecg_p2.val(1,1750:2550))
title('ECG for patient (Myocardial infarction)')
ylabel('x[n]')
xlabel('n')
subplot(3,1,2)
plot(cceps(ecg_p2.val(1,1750:2550)))
title('complex cepstrum for patient (Myocardial infarction)')
ylabel('x^[n]')
xlabel('n')
subplot(3,1,3)
plot(rceps(ecg_p2.val(1,1750:2550)))
title('real cepstrum for patient (Myocardial infarction)')
ylabel('c_x[n]')
xlabel('n')
figure;
subplot(3,1,1)
plot(ecg_h.val(1,1750:550+1750))
title('ECG for healthy control')
ylabel('x[n]')
xlabel('n')
subplot(3,1,2)
plot(cceps(ecg_h.val(1,1750:2550)))
title('complex cepstrum for healthy control')
ylabel('x^[n]')
xlabel('n')
subplot(3,1,3)
plot(rceps(ecg_h.val(1,1750:2550)))
title('real cepstrum for healthy control')
ylabel('c_x[n]')
xlabel('n')