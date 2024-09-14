%% our information
samples = 1000;

alpha1 = 2*pi*rand;
alpha2 = 2*pi*rand;
alpha3 = 2*pi*rand;

n = 0:samples-1;
x = 10*cos(0.1*pi*n + alpha1) + 20*cos(0.4*pi*n + alpha2) + 10*cos(0.8*pi*n + alpha3) + randn(1, samples);


%% part a

f = linspace(-pi, pi, 1000);
PSD = ones(size(f));
add_delta = @(psd, freq, amp) psd + amp * (f == freq);
PSD = add_delta(PSD, 0.1*pi, pi * 50);
PSD = add_delta(PSD, -0.1*pi, pi * 50);
PSD = add_delta(PSD, 0.4 * pi, pi * 200);
PSD = add_delta(PSD, -0.4 * pi, pi * 200);
PSD = add_delta(PSD, 0.8 * pi, pi * 50);
PSD = add_delta(PSD, -0.8 * pi, pi * 50);
figure;
plot(f, PSD);
title('Power Spectral Density (PSD)');
xlabel('Frequency');
ylabel('Power/Frequency');
hold on;
stem([0.1*pi, -0.1*pi, 0.4 * pi, -0.4 * pi, 0.8 * pi, -0.8 * pi], ...
    [pi * 50, pi * 50, pi * 200, pi * 200, pi * 50, pi * 50]);
xticks([-pi -0.8*pi -pi/2 -0.4*pi -0.1*pi 0 0.1*pi 0.4*pi pi/2 0.8*pi pi]);
xticklabels({'-\pi', '-0.8\pi', '-\pi/2' , '-0.4\pi', '-0.1\pi', '0', '0.1\pi', '0.4\pi', '\pi/2', '0.8\pi', '\pi'});



%% part b

estimated_corr = zeros(1,samples/2);
BT_PSD = zeros(1, 20000);
f1 = linspace(-pi, pi, 20000);

for i=1:samples/2
    for j = 1:(samples-i)
        estimated_corr(i) = x(j)*x(j+i) + estimated_corr(i);
    end
    estimated_corr(i) = estimated_corr(i) / samples;
end
for w=1:20000
    temp = 0;
    for j=1:(samples/2)
        temp = estimated_corr(j) * exp(-1i*f1(w)*j) + temp;
    end
    BT_PSD(w) = estimated_corr(1) + 2 * real(temp);
end

figure;
subplot(2,1,1);
plot(f1,abs(BT_PSD))
title('Estimate PSD by BT');
xlabel('Frequency');
ylabel('Power/Frequency');
xticks([-pi -0.8*pi -pi/2 -0.4*pi -0.1*pi 0 0.1*pi 0.4*pi pi/2 0.8*pi pi]);
xticklabels({'-\pi', '-0.8\pi', '-\pi/2' , '-0.4\pi', '-0.1\pi', '0', '0.1\pi', '0.4\pi', '\pi/2', '0.8\pi', '\pi'});

subplot(2,1,2);
plot(f, PSD);
hold on;
title('Exact PSD');
xlabel('Frequency');
ylabel('Power/Frequency');
stem([0.1*pi, -0.1*pi, 0.4 * pi, -0.4 * pi, 0.8 * pi, -0.8 * pi], ...
    [pi * 50, pi * 50, pi * 200, pi * 200, pi * 50, pi * 50]);
xticks([-pi -0.8*pi -pi/2 -0.4*pi -0.1*pi 0 0.1*pi 0.4*pi pi/2 0.8*pi pi]);
xticklabels({'-\pi', '-0.8\pi', '-\pi/2' , '-0.4\pi', '-0.1\pi', '0', '0.1\pi', '0.4\pi', '\pi/2', '0.8\pi', '\pi'});

%% part c

[pxx,f2] = periodogram(x,[],[],2*pi);

figure;
subplot(2,1,1);
plot(f2 , pxx); % Adjust frequency range to [-pi, pi]
hold on;
plot(-f2,pxx)
title('Estimate PSD by Periodogram');
xlabel('Frequency');
ylabel('Power/Frequency');
xticks([-pi -0.8*pi -pi/2 -0.4*pi -0.1*pi 0 0.1*pi 0.4*pi pi/2 0.8*pi pi]);
xticklabels({'-\pi', '-0.8\pi', '-\pi/2' , '-0.4\pi', '-0.1\pi', '0', '0.1\pi', '0.4\pi', '\pi/2', '0.8\pi', '\pi'});

subplot(2,1,2);
plot(f, PSD);
hold on;
stem([0.1*pi, -0.1*pi, 0.4 * pi, -0.4 * pi, 0.8 * pi, -0.8 * pi], ...
    [pi * 50, pi * 50, pi * 200, pi * 200, pi * 50, pi * 50]);
title('Exact PSD');
xlabel('Frequency');
ylabel('Power/Frequency');
xticks([-pi -0.8*pi -pi/2 -0.4*pi -0.1*pi 0 0.1*pi 0.4*pi pi/2 0.8*pi pi]);
xticklabels({'-\pi', '-0.8\pi', '-\pi/2' , '-0.4\pi', '-0.1\pi', '0', '0.1\pi', '0.4\pi', '\pi/2', '0.8\pi', '\pi'});





%% part d


[pxx_welch_10, f3] = pwelch(x, 100, 10, samples, 2*pi);
[pxx_welch_20, f4] = pwelch(x, 100, 20, samples, 2*pi);
[pxx_welch_30, f5] = pwelch(x, 100, 30, samples, 2*pi);

figure;
subplot(3,1,1);
plot(f3, pxx_welch_10); 
hold on;
plot(-f3,pxx_welch_10)
hold on;
plot(f, PSD);
hold on;
stem([0.1*pi, -0.1*pi, 0.4 * pi, -0.4 * pi, 0.8 * pi, -0.8 * pi], ...
    [pi * 50, pi * 50, pi * 200, pi * 200, pi * 50, pi * 50]);
xlabel('Frequency');
ylabel('Power/Frequency');
legend('Estimate PSD by Welch (noverlap = 10)','Exact PSD')
xticks([-pi -0.8*pi -pi/2 -0.4*pi -0.1*pi 0 0.1*pi 0.4*pi pi/2 0.8*pi pi]);
xticklabels({'-\pi', '-0.8\pi', '-\pi/2' , '-0.4\pi', '-0.1\pi', '0', '0.1\pi', '0.4\pi', '\pi/2', '0.8\pi', '\pi'});

%%%%%

subplot(3,1,2);
plot(f4 , pxx_welch_20); 
hold on;
plot(-f4,pxx_welch_20)
hold on;
plot(f, PSD);
hold on;
stem([0.1*pi, -0.1*pi, 0.4 * pi, -0.4 * pi, 0.8 * pi, -0.8 * pi], ...
    [pi * 50, pi * 50, pi * 200, pi * 200, pi * 50, pi * 50]);
xlabel('Frequency');
ylabel('Power/Frequency');
legend('Estimate PSD by Welch (noverlap = 20)','Exact PSD')
xticks([-pi -0.8*pi -pi/2 -0.4*pi -0.1*pi 0 0.1*pi 0.4*pi pi/2 0.8*pi pi]);
xticklabels({'-\pi', '-0.8\pi', '-\pi/2' , '-0.4\pi', '-0.1\pi', '0', '0.1\pi', '0.4\pi', '\pi/2', '0.8\pi', '\pi'});

%%%%%


subplot(3,1,3);
plot(f5, pxx_welch_30); 
hold on;
plot(-f5,pxx_welch_30)
hold on;
plot(f, PSD);
hold on;
stem([0.1*pi, -0.1*pi, 0.4 * pi, -0.4 * pi, 0.8 * pi, -0.8 * pi], ...
    [pi * 50, pi * 50, pi * 200, pi * 200, pi * 50, pi * 50]);
xlabel('Frequency');
ylabel('Power/Frequency');
legend('Estimate PSD by Welch (noverlap = 30)','Exact PSD')
xticks([-pi -0.8*pi -pi/2 -0.4*pi -0.1*pi 0 0.1*pi 0.4*pi pi/2 0.8*pi pi]);
xticklabels({'-\pi', '-0.8\pi', '-\pi/2' , '-0.4\pi', '-0.1\pi', '0', '0.1\pi', '0.4\pi', '\pi/2', '0.8\pi', '\pi'});





%% part e


R_x = xcorr(x,'biased');

% implementing levinson by myself and AIC cross valiadation
E = [];
K=[];
AIC = zeros(50,1);
temp = 0;
R_xx = R_x(1000:1999);
for i = 1:50
    if i == 1
        E(i) = R_xx(i);
    elseif i == 2
        k(i) = -R_xx(i) / E(i-1);
        a(i,i) = k(i);
        E(i) = (1 - k(i)^2) * E(i-1);
    else
        for j = 1:i-1
            temp = temp + a(j,i-1) * R_xx(i-j);
        end
        k(i) = -(R_xx(i) + temp) / E(i-1);
        a(i,i) = k(i);
        for j = 1:i-1
            a(j,i) = a(j,i-1) + k(i) * a(i-j,i-1);
        end
        E(i) = (1 - k(i)^2) * E(i-1);
    end
    AIC(i) = samples * log(E(i)) + 2 * (i);
    temp = 0;
end
[~, p_opt_AIC] = min(AIC);
p_opt_LD = 1;
disp('Optimal AR Order (p) by Levinson-Durbin:');
disp(p_opt_aic);
disp('Optimal AR Order (p) by AIC:');
disp(p_opt_AIC);

figure;
plot(1:50, AIC);
title('AIC (cross validation) by my levinson function');
xlabel('Model Order (p)');
ylabel('AIC');
disp('Optimal AR coeffs:');
disp(a(:,p_opt_LD));
disp('Optimal AR coeffs:');
disp(a(:,p_opt_AIC));
% implementing levinson by matlab function and AIC cross valiadation

e = zeros(51,1);
aic = zeros(50,1);
[al, e(1)] = levinson(R_xx, 0); % Order 0 model
for p = 1:50
    [al, e(p+1)] = levinson(R_xx, p);
    aic(p) = samples*log(e(p+1))+2*p;
end
[~, p_opt_aic] = min(aic);
p_opt_LD_m = 1;
disp('Optimal AR Order (p) by Levinson-Durbin:');
disp(p_opt_aic);
disp('Optimal AR Order (p) by AIC:');
disp(p_opt_aic);


disp('Optimal AR coeffs:');
disp(a(:,p_opt_LD_m));
disp('Optimal AR coeffs:');
disp(a(:,p_opt_aic));

figure;
plot(aic);
title('AIC (cross validation) by matlab function');
xlabel('Model Order (p)');
ylabel('AIC');


%% part f

bestOrder = 0;
bestAIC = Inf;
for q = 1:50
    try
        model = arima('MALags', 1:q, 'Constant', 0);
        fit = estimate(model, x', 'Display', 'off');
        [~,~,logL] = infer(fit, x');
        numParams = q;
        aic(i) = -2 * logL + 2 * numParams;
        if aic(i) < bestAIC
            bestOrder = q;
            bestAIC = aic(i);
            bestFit = fit;
        end
    catch
        continue;
    end
end

disp('Best MA order: ')
disp(num2str(bestOrder));
disp('MA coefficients:');
disp(cell2mat(bestFit.MA)');

figure;
plot(aic);
title('AIC (cross validation) fot MA');
xlabel('Model Order (p)');
ylabel('AIC');







