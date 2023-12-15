function [ Fs ] = FeatExtct( im_in,Params,NumLayers,SL1_features )




imdata_batch = im_in;
% padsize = floor((sqrt(size(Params{1}.W,1)) - 1) / 2);
% padval = (mean(mean(imdata_batch(:, [1,size(imdata_batch,2)])))...
%     + mean(mean(imdata_batch([1,size(imdata_batch,1)],:))))/2;
% imdata_batch = padarray(imdata_batch,[padsize,padsize]);
% imdata_batch = (0.6./sqrt(0.1)).*imdata_batch;
for j = 1 : NumLayers
    %     if ~Params{j}.pars.toplayer
%     if j ~=NumLayers
%         padsize = floor(((size(Params{j}.W,1)) - 1) / 2);
%         imdata_batch = padarray(imdata_batch,[padsize,padsize]);
%     end
    
    [Epool] = LayerOut((imdata_batch),Params{j});  % figure,display_image(Epool)
    if (j+1 <= NumLayers)
        if (strcmp(Params{j+1}.pars.layerType,'g'))
            %temp12 = whiten_image_new1(Epool1);
            imdata_batch = standard_im(Epool);
        else
            imdata_batch = Epool;%rescale(Epool,0.1,0.9);%standard_im
        end
    end
    f1 = Epool;%
    %     end
    if SL1_features(j)
        Fs{1,j} = f1;
    else
        Fs{1,j} = {};
    end
end
end
