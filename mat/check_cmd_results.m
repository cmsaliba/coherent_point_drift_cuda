% check_cmd_results.m
% Chris Saliba
% 2017/07/28

close all; clear; clc;

bn1 = importdata('pat1.txt');
bn2 = importdata('pat2.txt');
bn_transformed = importdata('pat_transformed.txt');

% bn1 = importdata('fem1.txt');
% bn2 = importdata('fem2.txt');
% bn_transformed = importdata('fem_transformed.txt');

% bn1 = importdata('sca1.txt');
% bn2 = importdata('sca2.txt');
% bn_transformed = importdata('sca_transformed.txt');

figure()
hold all
scatter3(bn1(:,1), bn1(:,2), bn1(:,3));
scatter3(bn2(:,1), bn2(:,2), bn2(:,3));
scatter3(bn_transformed(:,1), bn_transformed(:,2), bn_transformed(:,3));
legend('bone1', 'bone2', 'transformed')
axis equal