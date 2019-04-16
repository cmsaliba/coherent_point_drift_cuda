% example_custom_params.m
% Chris Saliba
% 2017/07/28

close all; clear; clc;

% load the bones

% femurs
% bone1 = load('fem1.mat');
% bone2 = load('fem2.mat');

% patellas
bone1 = load('pat1.mat');
bone2 = load('pat2.mat');

% scapulas
% bone1 = load('sca1.mat');
% bone2 = load('sca2.mat');

% run the CPD algorithm
tic
[T, C] = cpd_cuda(bone1.pts, bone2.pts, 0.1, 1, 1, 500, 1e-8);
toc

%%
% plot
figure('units','normalized','outerposition',[0.1 0.1 0.8 0.8])

subplot(1,3,1)
patch('Faces', bone1.con+1, 'Vertices', bone1.pts, 'FaceColor', [0 0 1], 'FaceAlpha', 0.5, 'EdgeAlpha', 0.5)
patch('Faces', bone2.con+1, 'Vertices', bone2.pts, 'FaceColor', [1 0 0], 'FaceAlpha', 0.5, 'EdgeAlpha', 0.5)
legend('X', 'Y')
axis equal
view([1 1 1])

subplot(1,3,2)
patch('Faces', bone1.con+1, 'Vertices', bone1.pts, 'FaceColor', [0 0 1], 'FaceAlpha', 0.5, 'EdgeAlpha', 0.5)
patch('Faces', bone2.con+1, 'Vertices', T, 'FaceColor', [1 0 0], 'FaceAlpha', 0.5, 'EdgeAlpha', 0.5)
legend('X', 'T')
axis equal
view([1 1 1])

subplot(1,3,3)
patch('Faces', bone1.con+1, 'Vertices', bone1.pts, 'FaceColor', [0 0 1], 'FaceAlpha', 0.5, 'EdgeAlpha', 0.5)
patch('Faces', bone2.con+1, 'Vertices', bone1.pts(C,:), 'FaceColor', [1 0 0], 'FaceAlpha', 0.5, 'EdgeAlpha', 0.5)
legend('X', 'X(C)')
axis equal
view([1 1 1])

