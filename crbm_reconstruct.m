function negdata = crbm_reconstruct(S, CRBM, pars)
if pars.BN
    temp = CRBM.Gamma ./ squeeze(sqrt(CRBM.posvark) + 1e-8)';
    for b = 1:pars.num_bases  % only for one channel visible layer
        S(:,:,b,:) = temp(1,b).*S(:,:,b,:);
    end
end
negdata2 = conve(S, CRBM.W, 0);
negdata = bsxfun(@plus,negdata2,reshape(CRBM.vbias_vec,[1,1,pars.numchannels]));
%     if strcmp(pars.layerType,'g')
%         negdata = negdata + (pars.std_gaussian^2) * randn(size(negdata));
%     end
switch pars.ActFun
    case 'sigm'
        if strcmp(pars.layerType,'b')
            negdata = (1/(pars.std_gaussian^2)).*(negdata);
            negdata = 1./(1 + exp(-negdata));
        end
    case 'probmax'
        if strcmp(pars.layerType,'b')
            negdata = (1/(pars.std_gaussian^2)).*(negdata);
            negdata = 1./(1 + exp(-negdata));
        end
    case 'max'
        if strcmp(pars.layerType,'b')
            negdata = (1/(pars.std_gaussian^2)).*(negdata);
            negdata = 1./(1 + exp(-negdata));
        end%         negdata = (1/(pars.std_gaussian^2)).*(negdata);
%         negdata = ReLU_N(negdata,1);
        
end
end
