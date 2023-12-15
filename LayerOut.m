function [HPc_out] = LayerOut(im_in1,params)
Spacing = params.pars.spacing;
padsize = (size(params.W,1) - 1) / 2;
im_in2 = padarray(im_in1,[padsize,padsize],'replicate');
im_in2 = trim_image_for_spacing_fixconv(im_in2, params.pars.ws, params.pars.spacing);
HPc_out = [];
for i = 1:size(im_in2,4)
    [poshidexp] = crbm_Forward(im_in2(:,:,:,i), params);
    switch params.pars.ActFun
        case 'probmax'
            [HPc] = crbm_sample_multrand_infer(poshidexp, Spacing);
        case 'max'
            [poshidprobs, ~] = feval(@sigmoidAct, poshidexp);
            n = 6;
            poshidprobs = ReLU_N(poshidprobs,n);
            HPc = convnet_maxpool(poshidprobs,Spacing);
        case 'sigm'
            [poshidprobs] = feval(@sigmoidAct,poshidexp);
            %             poshidprobs = standard_im(poshidprobs);
%             if params.pars.res
% %                 im_in1 = ReLU_N(im_in1,1);
%                 poshidprobs = bsxfun(@plus,poshidprobs,im_in1);
%             end
            HPc = convnet_maxpool(poshidprobs,Spacing);
%             HPc = rescale(HPc);
    end
    HPc_out(:,:,:,i) = HPc;
end
return

function [poshidexp_Ssh] = crbm_Forward(imdata, params)
poshidexp_Ssh = convs(imdata,params.W , 0);

if params.pars.BN
    [poshidexp_Ssh,poshidexp_N] = BN_inference(poshidexp_Ssh,params);
    poshidexp_Ssh = (1/(params.pars.std_gaussian^2)).*poshidexp_Ssh;
else
    for b = 1:params.pars.num_bases
        poshidexp_Ssh(:,:,b,:)  = (1/(params.pars.std_gaussian^2)).*(poshidexp_Ssh(:,:,b,:) + params.hbias_vec(b));% + temp1 (b);
    end
    poshidexp_N = 0;
end


return

function [poshidexp_Ssh,poshidexp_N] = BN_inference(poshidexp,params)
poshidexp_Ssh = zeros(size(poshidexp));
% Normalization
xmn = bsxfun(@minus, poshidexp,params.CRBM.posmuk);
poshidexp_N = bsxfun(@rdivide, xmn,sqrt(params.CRBM.posvark + eps));  % Normalize

for l = 1 : size(poshidexp,3)   % Scale and shift
    poshidexp_Ssh(:,:,l,:) = (params.CRBM.Gamma(1,l) .* poshidexp_N(:,:,l,:)) + params.CRBM.Beta(1,l);
end

return
