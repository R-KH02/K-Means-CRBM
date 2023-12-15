function [soft_x] = negsoft(dis, alpha)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
minM = min(dis, [], 2);
M = bsxfun(@minus, dis, minM);
exp_M = exp(-alpha.*(M));
SumM = sum(exp_M,2);

soft_x = bsxfun(@rdivide, exp_M, SumM);
end