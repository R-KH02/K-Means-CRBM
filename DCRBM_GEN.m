function [ferr, PAR, poshidprobs, poshidstates, negdata,CRBM] = ...
    DCRBM_GEN(imdata, train_y,CRBM, pars,PAR, bias_mode, l2reg,l1reg)

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do convolution/ get poshidprobs
[poshidexp,poshidexp_N,pars,CRBM] = crbm_inference_gen(imdata,train_y, CRBM, pars,true);
CRBM.poshidexp = poshidexp;
switch pars.ActFun
    case 'probmax'
        [poshidstates, poshidprobs] = crbm_sample_multrand2(poshidexp, pars.spacing);
    case 'max'
        [poshidstates, poshidprobs] = crbm_sample_max(poshidexp, pars);
    case 'sigm'
        [poshidprobs, gp] = sigmoidAct(poshidexp);
        [poshidprobs,indx] = convnet_maxpool(poshidprobs,pars.spacing);
        if pars.nneighbors ~= pars.num_bases
            rmapsize = size(poshidprobs);
            for i1 = 1 :size(poshidprobs,4)
                [~, sind] = sort(poshidprobs(:,:,:,i1), 3, 'descend');
                sind = sind(:,:,1:pars.nneighbors);
                sind = reshape(permute(sind,[3,1,2]),pars.nneighbors,rmapsize(1)*rmapsize(2));
                indpool = zeros(rmapsize(3),rmapsize(1)*rmapsize(2));
                for i = 1:size(indpool,2)
                    indpool(sind(:,i),i) = 1;
                end
                indpool = reshape(permute(indpool,[2,3,1]), size(poshidprobs(:,:,:,i1)));
                poshidprobs(:,:,:,i1) = poshidprobs(:,:,:,i1) .* indpool;
            end
        end
        CRBM.gp = gp .* indx + eps;
        if size(poshidprobs,4) > 1
            poshidprobs = expand(poshidprobs, [pars.spacing pars.spacing 1 1]);
        else
            poshidprobs = expand(poshidprobs, [pars.spacing pars.spacing 1]);
        end
        poshidprobs =poshidprobs .* indx;% make them spasre 0.25

        poshidstates = double(poshidprobs > rand(size(poshidprobs)));
    otherwise
        return;
end
%CRBM.AvgHidden = mean(poshidprobs,3);  % figure,montage(CRBM.hMaps)
CRBM.poshidprobs = poshidprobs;

% % posprods = zeros([size(CRBM.W),size(imdata,4)]);
% % for p1 = 1 : size(poshidprobs,4)
% % posprods(:,:,:,:,p1) = convs4(imdata(:,:,:,p1), poshidprobs(:,:,:,p1), 0);
% % end
% % poshidact = permute(sum(poshidprobs,[1,2]),[4 3 1 2]);
% % posvisact = permute(sum(sum(imdata,1),2),[4 3 1 2]);


posprods = convs4(imdata, poshidprobs, 0);
poshidact = reshape(sum(poshidprobs,[4,2,1]),[1,pars.num_bases]);
posvisact = reshape(sum(imdata,[4,2,1]),[1,pars.numchannels]);
posGamma = reshape(sum(poshidexp_N.*poshidprobs,[4 2 1]),[1,pars.num_bases]);

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
neghidstates = poshidstates;%poshidstates;%poshidprobs;%poshidstates;
for j=1:pars.K_CD  %% pars.K_CD-step contrastive divergence  works only for CD = 1
    negdata = crbm_reconstruct(neghidstates,CRBM, pars);
    if pars.classrbm
        labelProbs = crbm_reconstruct_y(neghidstates, CRBM);
        %         train_y
        labelStates = samplematrix ( labelProbs); %sample label vector
    else
        labelProbs = [];
        labelStates = [];
    end
    [neghidexp,neghidexp_N,pars,CRBM] = crbm_inference_gen(negdata, labelStates, CRBM, pars,false);
    switch pars.ActFun
        case 'probmax'
            [neghidstates, neghidprobs] = crbm_sample_multrand2(neghidexp, pars.spacing);%
        case 'max'
            [neghidstates, neghidprobs] = crbm_sample_max(neghidexp, pars);
        case 'sigm'
            [neghidprobs, ~] = sigmoidAct(neghidexp);
            neghidprobs = neghidprobs .* indx;
            %             if pars.nneighbors ~= pars.num_bases
            %                 neghidprobs(:,:,:,1) = neghidprobs(:,:,:,1) .* indpool;
            %             end
            neghidstates = double(neghidprobs > rand(size(neghidprobs)));
        otherwise
            return;
    end
end

% negprods = tirbm_vishidprod_fixconv(negdata, neghidprobs, pars.ws);
negprods = convs4(negdata, neghidprobs, 0);
neghidact = reshape(sum(neghidprobs,[4,2,1]),[1,pars.num_bases]);
negvisact = reshape(sum(negdata,[4,2,1]),[1,pars.numchannels]);
negGamma = reshape(sum(neghidexp_N.*neghidprobs,[4 1 2]),[1, pars.num_bases]);

% % negprods = zeros([size(CRBM.W),size(negdata,4)]);
% % for p1 = 1 : size(poshidprobs,4)
% % negprods(:,:,:,:,p1) = convs4(negdata(:,:,:,p1), neghidprobs(:,:,:,p1), 0);
% % end
% % neghidact = permute(sum(sum(neghidprobs,1),2),[4 3 1 2]);
% % negvisact =  permute(sum(sum(negdata,1),2),[4 3 1 2]);

ferr.ferr_neg = mean((imdata - negdata).^2,'all');
% monitor the learning
if 0
    figure(21), display_image((imdata));
    figure(22), display_image((negdata));

    figure(21), montage(rescale(imdata(:,:,:,1)));
    figure(22), montage(rescale(negdata(:,:,:,1)));
    figure(25), display_network(reshape(posprods,[pars.ws^2,pars.numchannels,pars.num_bases]));
    figure(26), display_network(reshape(negprods,[pars.ws^2,pars.numchannels,pars.num_bases]));
    figure(27), display_image(poshidstates);
    figure(28), display_image(neghidstates);
    figure(29), display_network(reshape(CRBM.W,[pars.ws^2,pars.numchannels,pars.num_bases]));
    plotting_hist
    figure, display_image(CRBM.hMaps)
    figure,display_image(poshidprobs);%(:,:,1:32))
end
%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numcases1 = size(poshidprobs,1)*size(poshidprobs,2)*size(poshidprobs,4); % w & hidden bias norm. factor
numcases2 = size(imdata,1).*size(imdata,2).*size(imdata,4); % visible bias norm. factor

if pars.BN > 0
    % other BN transformation related calculations
    term_pos = CRBM.Gamma ./ squeeze(sqrt(CRBM.posvark) + 1e-8)';
    term_neg = term_pos;%CRBM.Gamma ./ squeeze(sqrt(CRBM.negvark) + 1e-8)';% changed from negvark
end


if strcmp(bias_mode, 'simple')
    X = squeeze(sum(poshidprobs,[1,2,4])/size(poshidprobs,1)/size(poshidprobs,2)/size(poshidprobs,4))';
    if isempty(CRBM.runningavg_prob)
        CRBM.runningavg_prob =  X;%poshidact./numcases1;
    else
        %
        CRBM.runningavg_prob = pars.eta_sparsity*CRBM.runningavg_prob + (1-pars.eta_sparsity)*X;
    end
    dW = 0;
else
    error('wrong adjust_bias mode!');
end

if pars.BN > 0
    % Beta Update
    dbeta_total = (poshidact-neghidact)/numcases1;
    dgamma_total = (posGamma - negGamma)/numcases1;
    % W
    %     for l = 1: size(CRBM.Gamma,1)
    %         posprods(:,:,:,l) = bsxfun(@times,term_pos(1,l),posprods(:,:,:,l));
    %         negprods(:,:,:,l) = bsxfun(@times,term_neg(1,l),negprods(:,:,:,l));
    %     end
    % hidden biases
    poshidact = poshidact.* term_pos;
    neghidact = neghidact.* term_neg;
else
    dbeta_total = 0;
    dgamma_total = 0;
end


%%
% dW_total = (posprods-negprods)/numcases - l2reg*W - weightcost_l1*sign(W) - pars.pbias_lambda*dW;
dW_total1 = sum(posprods-negprods,5)/(size(poshidprobs,4))/(size(poshidprobs,1))/(size(poshidprobs,2));%.*ws^2);
dW_total2 = - l2reg*CRBM.W;
% dW_total3 = - pars.pbias_lb*dW;
dW_total4 = - l1reg*((CRBM.W>0)*2-1);
dW_total = dW_total1 + dW_total2 + dW_total4;%dW_total3 +
%
dh_total = sum(poshidact-neghidact,1)/size(poshidprobs,1)/size(poshidprobs,2)/size(poshidprobs,4);%/pars.ws;%ws^2;%
dv_total = sum(posvisact-negvisact,1)/size(imdata,1)/size(imdata,2)/size(imdata,4);%sum((posvisact-negvisact),1)/numcases2;
% if 0
%     fprintf('||W||=%g, ||dWprod|| = %g, ||dWl2|| = %g, ||dWsparse|| = %g\n', sqrt(sum(W(:).^2)), sqrt(sum(dW_total1(:).^2)), sqrt(sum(dW_total2(:).^2)), sqrt(sum(dW_total3(:).^2)));
% end

if pars.classrbm  == 1
    positive_phasey = poshidact' * train_y;
    negative_phasey = neghidact' * labelStates;
    du_total1 = (positive_phasey - negative_phasey) /numcases1;
    du_total2 = - l2reg*CRBM.U;
    du_total = du_total1 + du_total2;
    dd_total = sum(train_y - labelStates,1)';% - ;
else
    du_total = [];
    dd_total = [];
end

%%
PAR.dW_gen = dW_total;
PAR.dhbias_gen = dh_total;
PAR.dvbias_gen = dv_total;
%
PAR.dU = du_total;
PAR.ddbias = dd_total;
%
PAR.dGamma = dgamma_total;
PAR.dBeta = dbeta_total;

end


% % % % function [ferr, PAR, poshidprobs, poshidstates, negdata,CRBM] = ...
% % % %     DCRBM_GEN(imdata, train_y,CRBM, pars,PAR, bias_mode, l2reg,l1reg)
% % % %
% % % % %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % do convolution/ get poshidprobs
% % % % [poshidexp,pars,CRBM] = crbm_inference_gen(imdata,train_y, CRBM, pars);
% % % % [poshidstates, poshidprobs] = crbm_sample_multrand2(poshidexp, pars.spacing);
% % % % %
% % % % CRBM.AvgHidden = mean(poshidprobs,3);
% % % % CRBM.poshidprobs = poshidprobs;
% % % %
% % % % posprods = convs4(imdata, poshidprobs, 0);
% % % % poshidact = reshape(sum(poshidprobs,[4,2,1]),[1,pars.num_bases]);
% % % % posvisact = reshape(sum(imdata,[4,2,1]),[1,pars.numchannels]);
% % % %
% % % % %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % neghidstates = poshidstates;%poshidprobs;%poshidstates;
% % % % for j=1:pars.K_CD  %% pars.K_CD-step contrastive divergence
% % % %     negdata = crbm_reconstruct(neghidstates,CRBM, pars);
% % % %     if pars.classrbm
% % % %         labelProbs = crbm_reconstruct_y(neghidstates, CRBM);
% % % %         %   train_y
% % % %         labelStates = samplematrix ( labelProbs); %sample label vector
% % % %     else
% % % %         labelProbs = [];
% % % %         labelStates = [];
% % % %     end
% % % %     reconData = crbm_reconstruct(poshidprobs,CRBM, pars);
% % % %     [neghidexp,pars] = crbm_inference_gen(negdata, labelStates, CRBM, pars);
% % % %         [neghidstates, neghidprobs] = crbm_sample_multrand2(neghidexp, pars.spacing);%
% % % % end
% % % % % negprods = tirbm_vishidprod_fixconv(negdata, neghidprobs, pars.ws);
% % % % negprods = convs4(negdata, neghidprobs, 0);
% % % % neghidact = reshape(sum(neghidprobs,[4,2,1]),[1,pars.num_bases]);
% % % % negvisact = reshape(sum(negdata,[4,2,1]),[1,pars.numchannels]);
% % % %
% % % % ferr.ferr_neg = mean((imdata - negdata).^2,'all');
% % % % ferr.ferr_recon = mean((imdata - reconData).^2,'all');
% % % %
% % % % % monitor the learning
% % % % if 0
% % % %     figure(21), montage(rescale(imdata));
% % % %     figure(22), montage(rescale(negdata));
% % % %     figure(23), montage(rescale(reconData));
% % % %     %figure(24), montage(rescale(CRBM.W));
% % % %     figure(25), display_network(reshape(posprods,[pars.ws^2,pars.numchannels,pars.num_bases]));
% % % %     figure(26), display_network(reshape(negprods,[pars.ws^2,pars.numchannels,pars.num_bases]));
% % % %     figure(27), display_image(poshidstates);
% % % %     figure(28), display_image(neghidstates);
% % % %     figure(29), display_network(reshape(CRBM.W,[pars.ws^2,pars.numchannels,pars.num_bases]));
% % % %     plotting_hist
% % % %     figure, display_image(CRBM.hMaps)
% % % %     figure,display_image(poshidprobs);%(:,:,1:32))
% % % % end
% % % % %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % numcases1 = size(poshidprobs,1)*size(poshidprobs,2)*size(poshidprobs,4); % w & hidden bias norm. factor
% % % % numcases2 = size(imdata,1).*size(imdata,2).*size(imdata,4); % visible bias norm. factor
% % % % if strcmp(bias_mode, 'simple')
% % % %     X = squeeze(sum(poshidprobs,[1,2,4])/size(poshidprobs,1)/size(poshidprobs,2)/size(poshidprobs,4))';
% % % %     if isempty(CRBM.runningavg_prob)
% % % %         CRBM.runningavg_prob =  X;%poshidact./numcases1;
% % % %         %         CRBM.runningavg_exp = poshidexp;
% % % %     else
% % % %         %
% % % %         CRBM.runningavg_prob = pars.eta_sparsity*CRBM.runningavg_prob + (1-pars.eta_sparsity)*X;
% % % %         %         CRBM.runningavg_exp = pars.eta_sparsity*CRBM.runningavg_exp + (1-pars.eta_sparsity)*poshidexp;
% % % %     end
% % % %     dW = 0;
% % % % else
% % % %     error('wrong adjust_bias mode!');
% % % % end
% % % % %%
% % % % % dW_total = (posprods-negprods)/numcases - l2reg*W - weightcost_l1*sign(W) - pars.pbias_lambda*dW;
% % % % dW_total1 = (posprods-negprods)/(size(poshidprobs,4))/(size(poshidprobs,1))/(size(poshidprobs,2));%.*ws^2);
% % % % dW_total2 = - l2reg*CRBM.W;
% % % % % dW_total3 = - pars.pbias_lb*dW;
% % % % dW_total4 = - l1reg*((CRBM.W>0)*2-1);
% % % % dW_total = dW_total1 + dW_total2 + dW_total4;%dW_total3 +
% % % % %
% % % % dh_total = (poshidact-neghidact)/size(poshidprobs,1)/size(poshidprobs,2)/size(poshidprobs,4);%/pars.ws;%ws^2;%
% % % % dv_total = (posvisact-negvisact)/size(imdata,1)/size(imdata,2)/size(imdata,4);%sum((posvisact-negvisact),1)/numcases2;
% % % % % if 0
% % % % %     fprintf('||W||=%g, ||dWprod|| = %g, ||dWl2|| = %g, ||dWsparse|| = %g\n', sqrt(sum(W(:).^2)), sqrt(sum(dW_total1(:).^2)), sqrt(sum(dW_total2(:).^2)), sqrt(sum(dW_total3(:).^2)));
% % % % % end
% % % %
% % % % if pars.classrbm  == 1
% % % %     positive_phasey = poshidact' * train_y;
% % % %     negative_phasey = neghidact' * labelStates;
% % % %     du_total1 = (positive_phasey - negative_phasey) /numcases1;
% % % %     du_total2 = - l2reg*CRBM.U;
% % % %     du_total = du_total1 + du_total2;
% % % %     dd_total = sum(train_y - labelStates,1)';% - ;
% % % % else
% % % %     du_total = [];
% % % %     dd_total = [];
% % % % end
% % % %
% % % % %%
% % % % PAR.dW_gen = dW_total;
% % % % PAR.dhbias_gen = dh_total;
% % % % PAR.dvbias_gen = dv_total;
% % % % PAR.dU = du_total;
% % % % PAR.ddbias = dd_total;
% % % %
% % % %
% % % %
% % % % if 0
% % % %     figure(21), display_image((imdata));
% % % %     figure(22), display_image((negdata));
% % % %     figure(24), display_network(CRBM.W);
% % % %     figure(25), display_network(posprods);
% % % %     figure(26), display_network(negprods);
% % % %     figure(27), display_image(poshidstates);
% % % %     figure(28), display_image(neghidstates);
% % % %     plotting_hist
% % % % end
% % % % end
