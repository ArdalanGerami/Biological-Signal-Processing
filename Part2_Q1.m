clc; clear
%% inputs
a = 0.6;
b = 0.4;
u1_std = sqrt(12)*4;
u2_std = sqrt(12)*3;
v_std = sqrt(12)*2;
k = 100;
u1_a = u1_std * randn(1, k);
u2_a = u2_std * randn(1, k);
v_a = v_std * randn(1, k);
u1_b = u1_std * rand(1, k);
u2_b = u2_std * rand(1, k);
v_b = v_std * rand(1, k);
rho = 0.8;
x1 = randn(k, 1);
x2 = randn(k, 1);
R = [1 rho; rho 1];
L = chol(R, 'lower');
correlated_noises = L * [x1 x2]';
u1_c = correlated_noises(1, :);
u2_c = correlated_noises(2, :);
v_c = v_std * randn(1, k);
est_x0_handout = [0,0];
est_x0_random = [0.65,0.78];
%% part a 
[~,~,est_x_a,x1_a,x2_a, ~, ~, ~, ~] = kalmanmanf_part(a, b, u1_std, u2_std, v_std, u1_a, u2_a, v_a, est_x0_handout);
[~,~,est_x_da,x1_da,x2_da, ~, ~, ~, ~] = kalmanmanf_part(a, b, u1_std, u2_std, v_std, u1_a, u2_a, v_a, est_x0_random);
figure;
subplot(2,1,1)
plot(est_x_a(1,:))
hold on
plot(x1_a,'--')
legend('estimated X_1','real X_1')
title('state variables (gaussian u1 & u2) for X_1 - initial state = like example')
ylabel('X_1[n]')
xlabel('n')
MSE = mean((est_x_a(1,:)-x1_a).^2,2);
text(60,-30,['the MSE = ' , num2str(MSE)])
subplot(2,1,2)
plot(est_x_da(1,:))
hold on
plot(x1_da,'--')
legend('estimated X_1','real X_1')
title('state variables (gaussian u1 & u2) for X_1 - initial state = random')
ylabel('X_1[n]')
xlabel('n')
MSE = mean((est_x_da(1,:)-x1_da).^2,2);
text(60,-30,['the MSE = ' , num2str(MSE)])
% part d
figure;
subplot(2,1,1)
plot(est_x_a(2,:))
hold on
plot(x2_a,'--')
legend('estimated X_1','real X_1')
title('state variables (gaussian u1 & u2) for X_2 - initial state = like example')
ylabel('X_1[n]')
xlabel('n')
MSE = mean((est_x_a(2,:)-x2_a).^2,2);
text(60,-30,['the MSE = ' , num2str(MSE)])
subplot(2,1,2)
plot(est_x_da(2,:))
hold on
plot(x2_da,'--')
legend('estimated X_1','real X_1')
title('state variables (gaussian u1 & u2) for X_2 - initial state = random')
ylabel('X_1[n]')
xlabel('n')
MSE = mean((est_x_da(2,:)-x2_da).^2,2);
text(60,-30,['the MSE = ' , num2str(MSE)])
%% part b 
[~,~,est_x_b,x1_b,x2_b, ~, ~, ~, ~] = kalmanmanf_part(a, b, u1_std, u2_std, v_std, u1_b, u2_b, v_b, est_x0_handout);
[~,~,est_x_db,x1_db,x2_db, ~, ~, ~, ~] = kalmanmanf_part(a, b, u1_std, u2_std, v_std, u1_b, u2_b, v_b, est_x0_random);
figure;
subplot(2,1,1)
plot(est_x_b(1,:))
hold on
plot(x1_b,'--')
legend('estimated X_1','real X_1')
title('state variables (u1, u2 & v are not gaussian) for X_1 - initial state = like example')
ylabel('X_1[n]')
xlabel('n')
MSE = mean((est_x_b(1,:)-x1_b).^2,2);
text(60,5,['the MSE = ' , num2str(MSE)])
subplot(2,1,2)
plot(est_x_db(1,:))
hold on
plot(x1_db,'--')
legend('estimated X_1','real X_1')
title('state variables (u1, u2 & v are not gaussian) for X_1 - initial state = random')
ylabel('X_1[n]')
xlabel('n')
MSE = mean((est_x_db(1,:)-x1_db).^2,2);
text(60,5,['the MSE = ' , num2str(MSE)])
% 
figure;
subplot(2,1,1)
plot(est_x_b(2,:))
hold on
plot(x2_b,'--')
legend('estimated X_1','real X_1')
title('state variables (u1, u2 & v are not gaussian) for X_2 - initial state = like example')
ylabel('X_2[n]')
xlabel('n')
MSE = mean((est_x_b(2,:)-x2_b).^2,2);
text(60,5,['the MSE = ' , num2str(MSE)])
subplot(2,1,2)
plot(est_x_db(2,:))
hold on
plot(x2_db,'--')
legend('estimated X_1','real X_1')
title('state variables (u1, u2 & v are not gaussian) for X_2 - initial state = random')
ylabel('X_2[n]')
xlabel('n')
MSE = mean((est_x_db(2,:)-x2_db).^2,2);
text(60,5,['the MSE = ' , num2str(MSE)])
%% part c
[~,~,est_x_c,x1_c,x2_c, ~, ~, ~, ~] = kalmanmanf_partc(a, b, u1_std, u2_std, v_std, u1_c, u2_c, v_c, est_x0_handout,rho);
[~,~,est_x_dc,x1_dc,x2_dc, ~, ~, ~, ~] = kalmanmanf_partc(a, b, u1_std, u2_std, v_std, u1_c, u2_c, v_c, est_x0_random,rho);
figure;
subplot(2,1,1)
plot(est_x_c(1,:))
hold on
plot(x1_c,'--')
legend('estimated X_1','real X_1')
title('state variables (correlation between gaussian u1 & u2) for X_1 - initial state = like example')
ylabel('X_1[n]')
xlabel('n')
MSE = mean((est_x_c(1,:)-x1_c).^2,2);
text(60,-5,['the MSE = ' , num2str(MSE)])
subplot(2,1,2)
plot(est_x_dc(1,:))
hold on
plot(x1_dc,'--')
legend('estimated X_1','real X_1')
title('state variables (correlation between gaussian u1 & u2) for X_1 - initial state = random')
ylabel('X_1[n]')
xlabel('n')
MSE = mean((est_x_dc(1,:)-x1_dc).^2,2);
text(60,-5,['the MSE = ' , num2str(MSE)])
% 
figure;
subplot(2,1,1)
plot(est_x_c(2,:))
hold on
plot(x2_c,'--')
legend('estimated X_2','real X_2')
title('state variables (correlation between gaussian u1 & u2) for X_2 - initial state = like example')
ylabel('X_2[n]')
xlabel('n')
MSE = mean((est_x_c(2,:)-x2_c).^2,2);
text(60,-5*2,['the MSE = ' , num2str(MSE)])
subplot(2,1,2)
plot(est_x_dc(2,:))
hold on
plot(x2_dc,'--')
legend('estimated X_2','real X_2')
title('state variables (correlation between gaussian u1 & u2) for X_2 - initial state = random')
ylabel('X_2[n]')
xlabel('n')
MSE = mean((est_x_dc(2,:)-x2_dc).^2,2);
text(60,-5*2,['the MSE = ' , num2str(MSE)])
%% the function for kalman filter
function [estimated_z,real_output,est_x,x1,x2, est_xminus, est_p, est_pminus, est_G] = kalmanmanf_part(a, b, std1, std2, stdv, u1, u2 ,v ,est_x0)
    k = 100;
    H1 = tf([1], [1 -a], 1);
    H2 = tf([1], [1 -b], 1); 
    x1 = filter(H1.Numerator{1}, H1.Denominator{1}, u1);
    x2 = filter(H2.Numerator{1}, H2.Denominator{1}, x1 + u2);
    real_output = x2 + v;
    est_x = zeros(2, k);
    est_xminus = zeros(2, k);
    est_p = zeros(2, 2, k);
    est_pminus = zeros(2, 2, k);
    est_G = zeros(2, 1, k);
    est_p(:, :, 1) = [std1^2/(1-a^2), std1^2/((1-a^2)*(1-a*b)); std1^2/((1-a^2)*(1-a*b)), (std1^2+(1-a^2)*std2^2)/((1-a^2)*(1-b^2))];
    est_x(:, 1) = est_x0;
    est_x(:, 1) = [0.65; 0.78];
    F = [a,0;a,b];
    Q = [std1^2, std1^2; std1^2, std1^2 + std2^2];
    R = stdv^2;
    H = [0,1];
    estimated_z = zeros(1,k);
    estimated_z(1,1) = H*est_x(:,1) + v(1);
    for i = 2:k
        est_xminus(:, i) = F * est_x(:, i-1);
        est_pminus(:, :, i) = F * est_p(:, :, i-1) * F' + Q;
        Gk = est_pminus(:, :, i) * H' / (H * est_pminus(:, :, i) * H' + R);
        est_x(:, i) = est_xminus(:, i) + Gk * (real_output(i) - H * est_xminus(:, i));
        est_p(:, :, i) = est_pminus(:, :, i) - Gk * H * est_pminus(:, :, i);    
        est_G(:, :, i) = Gk;
        estimated_z(1,i) = H*est_x(:,i) + v(i);
    end
end

function [estimated_z,real_output,est_x,x1,x2, est_xminus, est_p, est_pminus, est_G] = kalmanmanf_partc(a, b, std1, std2, stdv, u1, u2 ,v ,est_x0,rho)
    k = 100;
    H1 = tf([1], [1 -a], 1);
    H2 = tf([1], [1 -b], 1); 
    x1 = filter(H1.Numerator{1}, H1.Denominator{1}, u1);
    x2 = filter(H2.Numerator{1}, H2.Denominator{1}, x1 + u2);
    real_output = x2 + v;
    est_x = zeros(2, k);
    est_xminus = zeros(2, k);
    est_p = zeros(2, 2, k);
    est_pminus = zeros(2, 2, k);
    est_G = zeros(2, 1, k);
    est_p(:, :, 1) = [std1^2/(1-a^2), std1*std2*rho/((1-a^2)*(1-a*b)); std1*std2*rho/((1-a^2)*(1-a*b)), (std1^2+std2^2)/((1-b^2))];
    est_x(:, 1) = est_x0;
    est_x(:, 1) = [0.65; 0.78];
    F = [a,0;a,b];
    Q = [std1^2, std1*std2*rho; std1*std2*rho, std1^2 + std2^2];
    R = stdv^2;
    H = [0,1];
    estimated_z = zeros(1,k);
    estimated_z(1,1) = H*est_x(:,1) + v(1);
    for i = 2:k
        est_xminus(:, i) = F * est_x(:, i-1);
        est_pminus(:, :, i) = F * est_p(:, :, i-1) * F' + Q;
        Gk = est_pminus(:, :, i) * H' / (H * est_pminus(:, :, i) * H' + R);
        est_x(:, i) = est_xminus(:, i) + Gk * (real_output(i) - H * est_xminus(:, i));
        est_p(:, :, i) = est_pminus(:, :, i) - Gk * H * est_pminus(:, :, i);    
        est_G(:, :, i) = Gk;
        estimated_z(1,i) = H*est_x(:,i) + v(i);
    end
end