%% a
ECG_dataset = load('test.mat');
fs = 360;
s = ECG_dataset.val;
t = 0:1/fs:(length(s)-1)/fs;
phi1 = 2*pi*rand;
phi2 = 2*pi*rand;
N1 = 20*cos(100*pi*t+phi1);
N2 = 20*cos(100*pi*t+phi2);
reference_signal = N1;
primary_signal = s + N2;
w = zeros(length(s),1);
y = zeros(length(s),1);
e = zeros(length(s),1);
mu = 0.000001;
p = y-w;
[e,y] = adaptivefilter(w,reference_signal,primary_signal,mu);

%% b

figure;

subplot(2,1,1)
plot(t,s)
title('ECG');
xlabel('time (sec)');
ylabel('amplitude');

subplot(2,1,2)
plot(t,e)
title('Estimated ECG');
xlabel('time (sec)');
ylabel('amplitude');

f = (-length(s)/2:length(s)/2-1)*(2*pi/length(s));
figure;
subplot(2,1,1)
plot(f, abs(fftshift(fft(s)))/length(s));
title('Magnitude Spectrum using FFT of ECG');
xlabel('Frequency (rad/sample)');
ylabel('Magnitude');
xticks([-pi  -pi/2  0  pi/2  pi]);
xticklabels({'-\pi',  '-\pi/2' , '0', '\pi/2', '\pi'});

subplot(2,1,2)
plot(f, abs(fftshift(fft(e)))/length(e))
title('Magnitude Spectrum using FFT of Estimated ECG');
xlabel('Frequency (rad/sample)');
ylabel('Magnitude');
xticks([-pi  -pi/2  0  pi/2  pi]);
xticklabels({'-\pi',  '-\pi/2' , '0', '\pi/2', '\pi'});

%% c

for i = 0:1000
    N1 = i*cos(100*pi*t+phi1);
    N2 = i*cos(100*pi*t+phi2);
    reference_signal = N1;
    primary_signal = s + N2;
    w = zeros(length(s),1);
    y = zeros(length(s),1);
    e = zeros(length(s),1);
    mu = 0.000001;
    [e,y] = adaptivefilter(w,reference_signal,primary_signal,mu);
    SNRin(i+1) = 10*log10(norm(s)^2/norm(N2)^2);
    SNRout(i+1) = 10*log10(norm(s)^2/(norm(e-s)^2));
    SNRimprovement(i+1) = SNRout(i+1) - SNRin(i+1);
end

figure;
plot(SNRin,SNRimprovement)
title('SNRimprovement vs. SNRin');
xlabel('SNRin');
ylabel('SNRimprovement');


%% part d


f = 50; 
z = tf('z', 1/fs); 
figure;
mu_values = [10 1 0.1 0.01 0.001 0.0001 0.00001 0.000001];
M = 50;
A = 20;
for i = 1:length(mu_values)
    mu = mu_values(i);
    H = (1 - 2*cos(2*pi*f/fs)*z^(-1)+z^(-2)) / ...
        (1 - 2*(1 - (mu*(M+1)*A^2)/4)*cos(2*pi*f/fs)*z^(-1) + (1 - mu*(M+1)*A^2)*z^(-2));
    subplot(3, 1, 1);
    hold on;
    [mag, phase, w] = bode(H);
    plot(w*fs/(2*pi), mag(:));
end
title('Frequency Response for Different Values of \mu');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
legend(arrayfun(@(x) ['\mu = ' num2str(x)], mu_values, 'UniformOutput', false));
grid on;

mu = 0.001;
M_values = 1:10;
A = 20;

for M = M_values
    H = (1 - 2*cos(2*pi*f/fs)*z^(-1)+z^(-2)) / ...
        (1 - 2*(1 - (mu*(M+1)*A^2)/4)*cos(2*pi*f/fs)*z^(-1) + (1 - mu*(M+1)*A^2)*z^(-2));
    subplot(3, 1, 2);
    hold on;
    [mag, phase, w] = bode(H);
    plot(w*fs/(2*pi), mag(:));
end
title('Frequency Response for Different Values of M');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
legend(arrayfun(@(x) ['M = ' num2str(x)], M_values, 'UniformOutput', false));
grid on;
mu = 0.001;
M = 50;
A_values = 0:10;
for A = A_values
    H = (1 - 2*cos(2*pi*f/fs)*z^(-1)+z^(-2)) / ...
        (1 - 2*(1 - (mu*(M+1)*A^2)/4)*cos(2*pi*f/fs)*z^(-1) + (1 - mu*(M+1)*A^2)*z^(-2));
    subplot(3, 1, 3);
    hold on;
    [mag, phase, w] = bode(H);
    plot(w*fs/(2*pi), mag(:));
end
title('Frequency Response for Different Values of A');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
legend(arrayfun(@(x) ['A = ' num2str(x)], A_values, 'UniformOutput', false));
grid on;
hold off;
%%%%%%%%%%%%%%%%%
mu = 0.0001;
A = 5;
M = 100;
figure;
subplot(3, 1, 1);
impulse(H);
title('Impulse Response');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
subplot(3, 1, 2);
[mag, phase, w] = bode(H, {0, pi});
mag = squeeze(mag);
w = squeeze(w);
plot(w*fs/(2*pi), 20*log10(mag));
title('Frequency Response');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;
subplot(3, 1, 3);
pzmap(H);
title('Poles and Zeros');


%% part e


t = 0:1/fs:(length(s)-1)/fs;
phi1 = 2*pi*rand;
phi2 = 2*pi*rand;
noise = rand(1,length(t));
N1 = 20*cos(100*pi*(t+noise)+phi1);
N2 = 20*cos(100*pi*(t+noise)+phi2);
reference_signal = N1;
primary_signal = s + N2;
w = zeros(length(s),1);
y = zeros(length(s),1);
e = zeros(length(s),1);
mu = 0.000001;
[e,y] = adaptivefilter(w,reference_signal,primary_signal,mu);

figure;

subplot(2,1,1)
plot(t,s)
title('ECG');
xlabel('time (sec)');
ylabel('amplitude');

subplot(2,1,2)
plot(t,e)
title('Estimated ECG');
xlabel('time (sec)');
ylabel('amplitude');


%% part f 

phi1 = 2*pi*rand;
phi2 = 2*pi*rand;
phi3 = 2*pi*rand;
N1 = 20*cos(100*pi*t+phi1);
N2 = 20*cos(100*pi*t+phi2);
reference_signal = N1;
primary_signal = s + N2;
w = zeros(length(s),1);
y = zeros(length(s),1);
e = zeros(length(s),1);
mu = 0.000001;
[eh1,yh1] = adaptivefilter(w,reference_signal,primary_signal,mu);

noise = rand(1,length(t));
N1 = 20*cos(100*pi*(t+noise)+phi1);
N2 = 20*cos(100*pi*(t+noise)+phi2);
reference_signal = N1;
primary_signal = s + N2;
w = zeros(length(s),1);
y = zeros(length(s),1);
e = zeros(length(s),1);
mu = 0.000001;
[eh,yh] = adaptivefilter(w,reference_signal,primary_signal,mu);

figure;
subplot(3,1,1)
plot(t,s)
title('ECG');
xlabel('time (sec)');
ylabel('amplitude');

subplot(3,1,2)
plot(t,eh1)
title('Estimated ECG');
xlabel('time (sec)');
ylabel('amplitude');

subplot(3,1,3)
plot(t,eh)
title('Estimated ECG (changing the sinusoidal frequency)');
xlabel('time (sec)');
ylabel('amplitude');



%% part g

phi1 = 2*pi*rand;
N2 = 20*cos(100*pi*t+phi2);
primary_signal = s + N2;
w = zeros(length(s),1);
mu = 0.000001;
for i = 1:15
    [eal(i), yal(i)] = ALEfilter(w, primary_signal, primary_signal, mu, i);
    SNRinal(i) = 10*log10(norm(s)^2/norm(N2)^2);
    SNRoutal(i) = 10*log10(norm(s)^2/(norm(e(i)-s)^2));
    SNRimprovemental(i) = SNRoutal(i) - SNRinal(i);
end
figure;
plot(SNRin,SNRimprovement)
title('SNRimprovement');
xlabel('sample');
ylabel('SNRimprovement');

%% Adaptive filter function :

function [e,y] = adaptivefilter(w,reference_signal,primary_signal,mu)
    for i=1:length(reference_signal)
        y = reference_signal*w;
        e = primary_signal - y;
        w = w + 2*mu*e*reference_signal';
    end
end
