clc; clear
ecg = load('ECG.mat');
train_num = 40;
test_num = 40;
%% part a
feature1 = zeros(1,train_num);
feature2 = zeros(1,train_num);
feature3 = zeros(1,train_num);
mean_train = zeros(1,train_num);
for i = 1:train_num
    if i<=9
        field_name = ['ECG0' num2str(i)];
    else
        field_name = ['ECG' num2str(i)];
    end
    mean_train(i) = mean(ecg.(field_name));
    lower_than_m= [];
    higher_than_m = [];
    k = 0;
    l = 0;
    for j = 1:length(ecg.(field_name))
        if ecg.(field_name)(j)<mean_train(i)
           k = k+1;
           lower_than_m(k) = ecg.(field_name)(j);
        else
           l = l+1;
           higher_than_m(l) = ecg.(field_name)(j);
        end
    end
    feature1(i) = mean(lower_than_m) / min(ecg.(field_name));
    feature2(i) = mean(lower_than_m)/var(lower_than_m);
    feature3(i) = (mean(higher_than_m)-mean(lower_than_m))/(max(ecg.(field_name))-min(ecg.(field_name)));
end
%% part b
class1_featurematrix = [feature1(1:20)' feature2(1:20)' feature3(1:20)'];
mean1_mat = mean(class1_featurematrix);
cov1_mat = cov(class1_featurematrix);
class2_featurematrix = [feature1(21:40)' feature2(21:40)' feature3(21:40)'];
mean2_mat = mean(class2_featurematrix);
cov2_mat = cov(class2_featurematrix);
cov_mat_train = cov([feature1(1:40)' feature2(1:40)' feature3(1:40)']);
x_range = linspace(0, 0.6, 40);
y_range = linspace(0, -0.1, 40);
z_range = linspace(0, 0.35, 40);
[X, Y, Z] = meshgrid(x_range, y_range, z_range);
D1 = zeros(size(X));
D2 = zeros(size(X));
x_flat = X(:);
y_flat = Y(:);
z_flat = Z(:);
for i = 1:length(x_flat)
    beta = [x_flat(i); y_flat(i); z_flat(i)];
    D1(i) = -0.5 * log(det(cov1_mat)) - 0.5 * (beta - mean1_mat')' * inv(cov1_mat) * (beta - mean1_mat');
    D2(i) = -0.5 * log(det(cov2_mat)) - 0.5 * (beta - mean2_mat')' * inv(cov2_mat) * (beta - mean2_mat');
end
D1 = reshape(D1, size(X));
D2 = reshape(D2, size(X));
Difference = D1 - D2;
figure;
isosurface(X, Y, Z, Difference, 0);
hold on;
scatter3(class1_featurematrix(:,1), class1_featurematrix(:,2), class1_featurematrix(:,3), 'r', 'filled');
hold on;
scatter3(class2_featurematrix(:,1), class2_featurematrix(:,2), class2_featurematrix(:,3), 'b', 'filled');
xlabel('Feature 1 (x)');
ylabel('Feature 2 (y)');
zlabel('Feature 3 (z)');
title('Bayes Decision Boundary');
grid on;
legend({'Decision Boundary', 'Class 1', 'Class 2'}, 'Location', 'best')
%% part c
[evc1, eva1] = eig(cov1_mat);
[evc2, eva2] = eig(cov2_mat);
[evc, eva] = eig(cov_mat_train);
u = evc(:,1:2);
u1 = evc1(:,1:2);  % Assuming these are the principal components
u2 = evc2(:,1:2);
class1_featurematrix_entropy = class1_featurematrix * u;
class2_featurematrix_entropy = class2_featurematrix * u;
mean1_mat_entropy = mean(class1_featurematrix_entropy)';
mean2_mat_entropy = mean(class2_featurematrix_entropy)';
cov1_mat_entropy = cov(class1_featurematrix_entropy);
cov2_mat_entropy = cov(class2_featurematrix_entropy);
x_range = linspace(-0.05, 0.03, 40);
y_range = linspace(-0.5, -0.2, 40);
[Xent, Yent] = meshgrid(x_range, y_range);
D1 = zeros(size(Xent));
D2 = zeros(size(Xent));
x_flat = Xent(:);
y_flat = Yent(:);
for i = 1:length(x_flat)
    beta = [x_flat(i); y_flat(i)];
    D1(i) = -0.5 * log(det(cov1_mat_entropy)) - 0.5 * (beta - mean1_mat_entropy)' * inv(cov1_mat_entropy) * (beta - mean1_mat_entropy);
    D2(i) = -0.5 * log(det(cov2_mat_entropy)) - 0.5 * (beta - mean2_mat_entropy)' * inv(cov2_mat_entropy) * (beta - mean2_mat_entropy);
end
D1 = reshape(D1, size(Xent));
D2 = reshape(D2, size(Xent));
Difference_ent = D1 - D2;
figure;
contour(Xent, Yent, Difference_ent, [0, 0], 'LineWidth', 2);
hold on;
scatter(class1_featurematrix_entropy(:, 1), class1_featurematrix_entropy(:, 2), 'r', 'filled');
hold on;
scatter(class2_featurematrix_entropy(:, 1), class2_featurematrix_entropy(:, 2), 'b', 'filled');
xlabel('Feature 1 (x)');
ylabel('Feature 2 (y)');
title('Decision Boundary for Entropy');
grid on;
legend({'Decision Boundary', 'Class 1', 'Class 2'}, 'Location', 'best');
%% part d

class1_featurematrix = [feature1(1:20)' feature2(1:20)' feature3(1:20)'];
class2_featurematrix = [feature1(21:40)' feature2(21:40)' feature3(21:40)'];
mean1 = mean(class1_featurematrix)';
mean2 = mean(class2_featurematrix)';
S_W = cov(class1_featurematrix) + cov(class2_featurematrix);
mean_diff = mean1 - mean2;
S_B = mean_diff * mean_diff';
[V, D] = eig(S_W \ S_B);
[~, idx] = max(diag(D));
w = V(:, idx);
projected_class1 = class1_featurematrix * w;
projected_class2 = class2_featurematrix * w;
mean_proj1 = mean(projected_class1);
mean_proj2 = mean(projected_class2);
var_proj1 = var(projected_class1);
var_proj2 = var(projected_class2);
pooled_variance = ((length(projected_class1) - 1) * var_proj1 + (length(projected_class2) - 1) * var_proj2) / (length(projected_class1) + length(projected_class2) - 2);
decision_boundary = (mean_proj1 + mean_proj2) / 2;
figure;
hold on;
scatter(projected_class1, zeros(size(projected_class1)), 'r', 'filled');
scatter(projected_class2, zeros(size(projected_class2)), 'b', 'filled');
plot([decision_boundary decision_boundary], [-0.5 0.5], 'k--', 'LineWidth', 2);
xlabel('Projected Feature');
ylabel('Value');
title('Fisher Discriminant Analysis (3D to 1D) with Decision Boundary');
legend({'Class 1', 'Class 2', 'Decision Boundary'}, 'Location', 'best');
grid on;
figure;
plot([decision_boundary decision_boundary], [-0.5 0.5], 'k--', 'LineWidth', 2);
hold on;
scatter(class1_featurematrix_entropy(:, 1), class1_featurematrix_entropy(:, 2), 'r', 'filled');
hold on;
scatter(class2_featurematrix_entropy(:, 1), class2_featurematrix_entropy(:, 2), 'b', 'filled');
xlabel('Feature 1');
ylabel('Feature 2');
title('Decision Boundary for two dimentional feature space');
grid on;
legend({'Decision Boundary', 'Class 1', 'Class 2'}, 'Location', 'best')
%% part e
feature1test = zeros(1,test_num);
feature2test  = zeros(1,test_num);
feature3test  = zeros(1,test_num);
mean_test = zeros(1,test_num);
for i = 41:80
    field_name = ['ECG' num2str(i)];
    mean_test(i-40) = mean(ecg.(field_name));
    lower_than_mtest= [];
    higher_than_mtest = [];
    k = 0;
    l = 0;
    for j = 1:length(ecg.(field_name))
        if ecg.(field_name)(j)<mean_test(i-40)
           k = k+1;
           lower_than_mtest(k) = ecg.(field_name)(j);
        else
           l = l+1;
           higher_than_mtest(l) = ecg.(field_name)(j);
        end
    end
    feature1test (i-40) = mean(lower_than_mtest) / min(ecg.(field_name));
    feature2test (i-40) = mean(lower_than_mtest)/var(lower_than_mtest);
    feature3test (i-40) = (mean(higher_than_mtest)-mean(lower_than_mtest))/(max(ecg.(field_name))-min(ecg.(field_name)));
end
class1test_featurematrix = [feature1test(1:20)' feature2test(1:20)' feature3test(1:20)'];
class2test_featurematrix = [feature1test(21:40)' feature2test(21:40)' feature3test(21:40)'];
cov1_mat_test = cov(class1test_featurematrix);
cov2_mat_test = cov(class2test_featurematrix);
cov_mat_test = cov([feature1test(1:40)' feature2test(1:40)' feature3test(1:40)']);
% for part b
labels_class1_mah = zeros(1,20);
labels_class2_mah = zeros(1,20);
labels_class1_o = zeros(1,20);
labels_class2_o = zeros(1,20);
labels_class1_bayes = zeros(1,20);
labels_class2_bayes = zeros(1,20);
temp_o = 0;
temp_mah = 0;
temp_bayes = 0;
for i = 1:20
    d1_mah = (class1test_featurematrix(i,1:3)-mean1_mat)*inv(cov1_mat)*(class1test_featurematrix(i,1:3)-mean1_mat)';
    d2_mah = (class1test_featurematrix(i,1:3)-mean2_mat)*inv(cov2_mat)*(class1test_featurematrix(i,1:3)-mean2_mat)';
    if d1_mah>d2_mah
        labels_class1_mah(1,i) = 2;
        temp_mah = temp_mah +1;
    else
        labels_class1_mah(1,i) = 1;
    end
    d1_mah = (class2test_featurematrix(i,1:3)-mean1_mat)*inv(cov1_mat)*(class2test_featurematrix(i,1:3)-mean1_mat)';
    d2_mah = (class2test_featurematrix(i,1:3)-mean2_mat)*inv(cov2_mat)*(class2test_featurematrix(i,1:3)-mean2_mat)';
    if d1_mah>d2_mah
        labels_class2_mah(1,i) = 2;
    else
        labels_class2_mah(1,i) = 1;
        temp_mah = temp_mah +1;
    end
end
for i = 1:20
    d1_o = (class1test_featurematrix(i,1:3)-mean1_mat)*(class1test_featurematrix(i,1:3)-mean1_mat)';
    d2_o = (class1test_featurematrix(i,1:3)-mean2_mat)*(class1test_featurematrix(i,1:3)-mean2_mat)';
    if d1_o>d2_o
        labels_class1_o(1,i) = 2;
        temp_o = temp_o +1;
    else
        labels_class1_o(1,i) = 1;
    end
    d1_o = (class2test_featurematrix(i,1:3)-mean1_mat)*(class2test_featurematrix(i,1:3)-mean1_mat)';
    d2_o = (class2test_featurematrix(i,1:3)-mean2_mat)*(class2test_featurematrix(i,1:3)-mean2_mat)';
    if d1_o>d2_o
        labels_class2_o(1,i) = 2;
    else
        labels_class2_o(1,i) = 1;
        temp_o = temp_o +1;
    end
end
for i = 1:20
    d1_o = +0.5*log(det(cov1_mat))+0.5*(class1test_featurematrix(i,1:3)-mean1_mat)*inv(cov1_mat)*(class1test_featurematrix(i,1:3)-mean1_mat)';
    d2_o = +0.5*log(det(cov2_mat))+0.5*(class1test_featurematrix(i,1:3)-mean2_mat)*inv(cov2_mat)*(class1test_featurematrix(i,1:3)-mean2_mat)';
    if d1_o>d2_o
        labels_class1_bayes(1,i) = 2;
        temp_bayes = temp_bayes +1;
    else
        labels_class1_bayes(1,i) = 1;
    end
    d1_o = +0.5*log(det(cov1_mat))+0.5*(class2test_featurematrix(i,1:3)-mean1_mat)*inv(cov1_mat)*(class2test_featurematrix(i,1:3)-mean1_mat)';
    d2_o = +0.5*log(det(cov2_mat))+0.5*(class2test_featurematrix(i,1:3)-mean2_mat)*inv(cov2_mat)*(class2test_featurematrix(i,1:3)-mean2_mat)';
    if d1_o>d2_o
        labels_class2_bayes(1,i) = 2;
    else
        labels_class2_bayes(1,i) = 1;
        temp_bayes = temp_bayes +1;
    end
end
disp('accuracy by mahlanobis (part b) :')
disp((40-temp_o)/40*100)
disp('accuracy by Euclidean (part b) :')
disp((40-temp_mah)/40*100)
disp('accuracy by bayes decision boundary (part b) :')
disp((40-temp_bayes)/40*100)
disp('-------------------------------------------------------------------')
% for part c
[evc1_test_b, eva1_test_b] = eig(cov1_mat_test);
[evc2_test_b, eva2_test_b] = eig(cov2_mat_test);
[evc_test_b, eva_test_b] = eig(cov_mat_test);
u1_test_b = evc1_test_b(:,1:2);  
u2_test_b = evc2_test_b(:,1:2);
u_test_b = evc_test_b(:,1:2);
class1_featurematrix_test_entropy = class1_featurematrix * u_test_b;
class2_featurematrix_test_entropy = class2_featurematrix * u_test_b;
labels_class1_mah_ent = zeros(1,20);
labels_class2_mah_ent = zeros(1,20);
labels_class1_o_ent = zeros(1,20);
labels_class2_o_ent = zeros(1,20);
labels_class1_bayes_ent = zeros(1,20);
labels_class2_bayes_ent = zeros(1,20);
temp_o = 0;
temp_mah = 0;
temp_bayes = 0;
for i = 1:20
    d1_mah = (class1_featurematrix_test_entropy(i,1:2)-mean1_mat_entropy)*inv(cov1_mat_entropy)*(class1_featurematrix_test_entropy(i,1:2)-mean1_mat_entropy)';
    d2_mah = (class1_featurematrix_test_entropy(i,1:2)-mean2_mat_entropy)*inv(cov2_mat_entropy)*(class1_featurematrix_test_entropy(i,1:2)-mean2_mat_entropy)';
    if d1_mah>d2_mah
        labels_class1_mah_ent(1,i) = 2;
        temp_mah = temp_mah +1;
    else
        labels_class2_mah_ent(1,i) = 1;
    end
    d1_mah = (class2_featurematrix_test_entropy(i,1:2)-mean1_mat_entropy)*inv(cov1_mat_entropy)*(class2_featurematrix_test_entropy(i,1:2)-mean1_mat_entropy)';
    d2_mah = (class2_featurematrix_test_entropy(i,1:2)-mean2_mat_entropy)*inv(cov2_mat_entropy)*(class2_featurematrix_test_entropy(i,1:2)-mean2_mat_entropy)';
    if d1_mah>d2_mah
        labels_class1_mah_ent(1,i) = 2;
    else
        labels_class2_mah_ent(1,i) = 1;
        temp_mah = temp_mah +1;
    end
end
for i = 1:20
    d1_o = (class1_featurematrix_test_entropy(i,1:2)-mean1_mat_entropy)*(class1_featurematrix_test_entropy(i,1:2)-mean1_mat_entropy)';
    d2_o = (class1_featurematrix_test_entropy(i,1:2)-mean2_mat_entropy)*(class1_featurematrix_test_entropy(i,1:2)-mean2_mat_entropy)';
    if d1_o>d2_o
        labels_class1_o_ent(1,i) = 2;
        temp_o = temp_o +1;
    else
        labels_class1_o_ent(1,i) = 1;
    end
    d1_o = (class2_featurematrix_test_entropy(i,1:2)-mean1_mat_entropy)*(class2_featurematrix_test_entropy(i,1:2)-mean1_mat_entropy)';
    d2_o = (class2_featurematrix_test_entropy(i,1:2)-mean2_mat_entropy)*(class2_featurematrix_test_entropy(i,1:2)-mean2_mat_entropy)';
    if d1_o>d2_o
        labels_class2_o_ent(1,i) = 2;
    else
        labels_class2_o_ent(1,i) = 1;
        temp_o = temp_o +1;
    end
end
for i = 1:20
    d1_mah = 0.5*log(det(cov1_mat_entropy))+0.5*(class1_featurematrix_test_entropy(i,1:2)-mean1_mat_entropy)*inv(cov1_mat_entropy)*(class1_featurematrix_test_entropy(i,1:2)-mean1_mat_entropy)';
    d2_mah = 0.5*log(det(cov2_mat_entropy))+0.5*(class1_featurematrix_test_entropy(i,1:2)-mean2_mat_entropy)*inv(cov2_mat_entropy)*(class1_featurematrix_test_entropy(i,1:2)-mean2_mat_entropy)';
    if d1_mah>d2_mah
        labels_class1_bayes_ent(1,i) = 2;
        temp_bayes = temp_bayes +1;
    else
        labels_class1_bayes_ent(1,i) = 1;
    end
    d1_mah = 0.5*log(det(cov1_mat_entropy))+0.5*(class2_featurematrix_test_entropy(i,1:2)-mean1_mat_entropy)*inv(cov1_mat_entropy)*(class2_featurematrix_test_entropy(i,1:2)-mean1_mat_entropy)';
    d2_mah = 0.5*log(det(cov2_mat_entropy))+0.5*(class2_featurematrix_test_entropy(i,1:2)-mean2_mat_entropy)*inv(cov2_mat_entropy)*(class2_featurematrix_test_entropy(i,1:2)-mean2_mat_entropy)';
    if d1_mah>d2_mah
        labels_class2_bayes_ent(1,i) = 2;
    else
        labels_class2_mah_ent(1,i) = 1;
        temp_bayes = temp_bayes +1;
    end
end
disp('accuracy by mahlanobis (part c) :')
disp((40-temp_mah)/40*100)
disp('accuracy by Euclidean (part c) :')
disp((40-temp_o)/40*100)
disp('accuracy by bayes decision boundary (part c) :')
disp((40-temp_bayes)/40*100)
disp('-------------------------------------------------------------------')
% for part d
mean1_test_fisher = mean(class1test_featurematrix)';
mean2_test_fisher = mean(class2test_featurematrix)';
S_W_test = cov(class1test_featurematrix) + cov(class2test_featurematrix);
mean_diff = mean1_test_fisher - mean2_test_fisher;
S_B = mean_diff * mean_diff';
[V_test, D_test] = eig(S_W_test \ S_B);
[~, idx_test] = max(diag(D_test));
w_test = V_test(:, idx_test);
projected_class1_test = class1test_featurematrix * w_test;
projected_class2_test = class2test_featurematrix * w_test;
mean_proj1_test = mean(projected_class1_test);
mean_proj2_test = mean(projected_class2_test);
var_proj1_test = var(projected_class1_test);
var_proj2_test = var(projected_class2_test);
labels_class1_mah_fisher = zeros(1,20);
labels_class2_mah_fisher = zeros(1,20);
labels_class1_o_fisher = zeros(1,20);
labels_class2_o_fisher = zeros(1,20);
labels_class1_o_bayes = zeros(1,20);
labels_class2_o_bayes = zeros(1,20);
temp_o = 0;
temp_mah = 0;
temp_bayes = 0;
for i = 1:20
    d1_mah = (projected_class1_test(i)-mean_proj1_test)*inv(var_proj1_test)*(projected_class1_test(i)-mean_proj1_test)';
    d2_mah = (projected_class1_test(i)-mean_proj2_test)*inv(var_proj2_test)*(projected_class1_test(i)-mean_proj2_test)';
    if d1_mah>d2_mah
        labels_class1_mah_fisher(1,i) = 2;
        temp_mah = temp_mah +1;
    else
        labels_class2_mah_fisher(1,i) = 1;
    end
    d1_mah = (projected_class2_test(i)-mean_proj1_test)*inv(var_proj1_test)*(projected_class2_test(i)-mean_proj1_test)';
    d2_mah = (projected_class2_test(i)-mean_proj2_test)*inv(var_proj2_test)*(projected_class2_test(i)-mean_proj2_test)';
    if d1_mah>d2_mah
        labels_class1_mah_fisher(1,i) = 2;
    else
        labels_class2_mah_fisher(1,i) = 1;
        temp_mah = temp_mah +1;
    end
end
for i = 1:20
    d1_o = (projected_class1_test(i)-mean_proj1_test)*(projected_class1_test(i)-mean_proj1_test)';
    d2_o = (projected_class1_test(i)-mean_proj2_test)*(projected_class1_test(i)-mean_proj2_test)';
    if d1_o>d2_o
        labels_class1_o_fisher(1,i) = 2;
        temp_o = temp_o +1;
    else
        labels_class1_o_fisher(1,i) = 1;
    end
    d1_o = (projected_class2_test(i)-mean_proj1_test)*(projected_class2_test(i)-mean_proj1_test)';
    d2_o = (projected_class2_test(i)-mean_proj2_test)*(projected_class2_test(i)-mean_proj2_test)';
    if d1_o>d2_o
        labels_class2_o_fisher(1,i) = 2;
    else
        labels_class2_o_fisher(1,i) = 1;
        temp_o = temp_o +1;
    end
end
for i = 1:20
    d1_mah = 0.5*log(det(var_proj1_test))+0.5*(projected_class1_test(i)-mean_proj1_test)*inv(var_proj1_test)*(projected_class1_test(i)-mean_proj1_test)';
    d2_mah = 0.5*log(det(var_proj2_test))+0.5*(projected_class1_test(i)-mean_proj2_test)*inv(var_proj2_test)*(projected_class1_test(i)-mean_proj2_test)';
    if d1_mah>d2_mah
        labels_class1_mah_fisher(1,i) = 2;
        temp_bayes = temp_bayes +1;
    else
        labels_class2_mah_fisher(1,i) = 1;
    end
    d1_mah = 0.5*log(det(var_proj1_test))+0.5*(projected_class2_test(i)-mean_proj1_test)*inv(var_proj1_test)*(projected_class2_test(i)-mean_proj1_test)';
    d2_mah = 0.5*log(det(var_proj2_test))+0.5*(projected_class2_test(i)-mean_proj2_test)*inv(var_proj2_test)*(projected_class2_test(i)-mean_proj2_test)';
    if d1_mah>d2_mah
        labels_class1_mah_fisher(1,i) = 2;
    else
        labels_class2_mah_fisher(1,i) = 1;
        temp_bayes = temp_bayes +1;
    end
end
disp('accuracy by mahlanobis (part d) :')
disp((40-temp_o)/40*100)
disp('accuracy by Euclidean (part d) :')
disp((40-temp_mah)/40*100)
disp('accuracy by bayes decision boundary (part d) :')
disp((40-temp_bayes)/40*100)
disp('-------------------------------------------------------------------')