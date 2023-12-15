function [poshidexp_out] = crbm_inference(imdata, W, hbias_vec, pars)

poshidexp2 = convs(imdata, W, 0);
% if posphase
%     CRBM.WeightDerivativ = mean(weightDerivative(poshidexp2,CRBM),5);
% end
for b = 1:pars.num_bases
    poshidexp2(:,:,b,:)  = poshidexp2(:,:,b,:) + hbias_vec(b);
end

poshidexp2 = (1/(pars.std_gaussian^2)).* poshidexp2;

poshidexp_out =  poshidexp2;

end