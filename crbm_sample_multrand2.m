function [H_out, HP_out, Hc_out, HPc_out, HP1_out] = crbm_sample_multrand2(poshidexp_batch, spacing)


for t1 = 1:size(poshidexp_batch,4)
    poshidexp = poshidexp_batch(:,:,:,t1);
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
    S = S1';
    P = P1';
    P2 = P2';
    clear S1 P1 sumP
    
    % convert back to original sized matrix
    H = zeros(size(poshidexp));
    HP = zeros(size(poshidexp));
    HP1 = zeros(size(poshidexp));
    for c = 1:spacing
        for r = 1:spacing
            H(r:spacing:end, c:spacing:end, :) = reshape(S((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
            HP(r:spacing:end, c:spacing:end, :) = reshape(P((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
            HP1(r:spacing:end, c:spacing:end, :) = reshape(P2((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
        end
    end
    
    if nargout >2
        Sc = sum(S(1:end-1,:),1);
        Pc = sum(P(1:end-1,:),1);
        Hc = reshape(Sc, [size(poshidexp,1)/spacing,size(poshidexp,2)/spacing,size(poshidexp,3)]);
        HPc = reshape(Pc, [size(poshidexp,1)/spacing,size(poshidexp,2)/spacing,size(poshidexp,3)]);
    end
    
    H_out(:,:,:,t1) = H;
    HP_out(:,:,:,t1) = HP;
    if nargout > 2
        Hc_out(:,:,:,t1) = Hc;
        HPc_out(:,:,:,t1) = HPc;
    end
    HP1_out(:,:,:,t1) = HP1;
end
return