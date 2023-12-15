function [HPc,HP] = crbm_sample_multrand_infer(poshidexp, spacing)
% if ~exist('spacing','var')
%     spacing = params.spacing;
% end

% poshidexp is 3d array
poshidprobs_mult = zeros(spacing^2+1, size(poshidexp,1)*size(poshidexp,2)*size(poshidexp,3)/spacing^2);
poshidprobs_mult(end,:) = 0;

for c = 1:spacing
    for r = 1:spacing
        temp = poshidexp(r:spacing:end, c:spacing:end, :);
        poshidprobs_mult((c-1)*spacing+r,:) = temp(:);
    end
end

% substract from max exponent to make values numerically more stable
poshidprobs_mult = bsxfun(@minus, poshidprobs_mult, max(poshidprobs_mult,[],1));
poshidprobs_mult = exp(poshidprobs_mult);

[S1, P1,sumP] = multrand2(poshidprobs_mult');
P2 = bsxfun(@rdivide,P1,sumP);
P = P1';
clear S1 P1 sumP

% convert back to original sized matrix
HP = zeros(size(poshidexp));
for c = 1:spacing
    for r = 1:spacing
        HP(r:spacing:end, c:spacing:end, :) = reshape(P((c-1)*spacing+r,:), [size(HP,1)/spacing, size(HP,2)/spacing, size(HP,3)]);
    end
end

    Pc = sum(P(1:end-1,:),1);
    HPc = reshape(Pc, [size(poshidexp,1)/spacing,size(poshidexp,2)/spacing,size(poshidexp,3)]);


return