function [ Fs ] = FeatExtct_1( im_in,Params,NumLayers )




imdata_batch = im_in;

for j = 1 : NumLayers

    [Epool] = LayerOut_inside((imdata_batch),Params{j},(NumLayers==j));  % figure,display_image(Epool)
    if (j+1 <= NumLayers)
        if (strcmp(Params{j+1}.pars.layerType,'g'))

            imdata_batch = standard_im(Epool);
        else
            imdata_batch = Epool;%rescale(Epool,0.1,0.9);%standard_im
        end
    end
    f1 = Epool;
    if j == NumLayers
        Fs = f1;
    end
end
end


function [HPc_out,Map_out] = LayerOut_inside(im_in1,params, last_layer)
Spacing = params.pars.spacing;
padsize = (size(params.W,1) - 1) / 2;
im_in2 = padarray(im_in1,[padsize,padsize],'replicate');
im_in2 = trim_image_for_spacing_fixconv(im_in2, params.pars.ws, params.pars.spacing);
Map_out = [];
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
            [HPc, map] = convnet_maxpool(poshidprobs,Spacing);
            %             HPc = rescale(HPc);
    end
    if last_layer
        HPc_out(:,:,:,i) = poshidprobs;
        Map_out(:,:,:,i) = [];
    else
        HPc_out(:,:,:,i) = HPc;
        Map_out(:,:,:,i) = map;
    end
end
end

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
end



function [poshidexp_Ssh,poshidexp_N] = BN_inference(poshidexp,params)
poshidexp_Ssh = zeros(size(poshidexp));
% Normalization
xmn = bsxfun(@minus, poshidexp,params.CRBM.posmuk);
poshidexp_N = bsxfun(@rdivide, xmn,sqrt(params.CRBM.posvark + eps));  % Normalize

for l = 1 : size(poshidexp,3)   % Scale and shift
    poshidexp_Ssh(:,:,l,:) = (params.CRBM.Gamma(1,l) .* poshidexp_N(:,:,l,:)) + params.CRBM.Beta(1,l);
end

end



