function [S, P,gradP] = multrand2(P)
% P is 2-d matrix: 2nd dimension is # of choices

sumP = sum(P,2);
gradP = bsxfun(@rdivide,P,sumP.^2);
P = bsxfun(@rdivide,P,sumP);


cumP = cumsum(P,2);
unifrnd = rand(size(P,1),1);
temp = bsxfun(@gt,cumP,unifrnd);
Sindx = diff(temp,1,2);
S = zeros(size(P));
S(:,1) = 1-sum(Sindx,2);
S(:,2:end) = Sindx;

return;