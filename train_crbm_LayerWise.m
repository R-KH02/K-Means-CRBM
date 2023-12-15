function params = train_crbm_LayerWise(opt,Params,LayerNum,DataSet)


BN = opt{LayerNum}.BN;
res = opt{LayerNum}.res;
ActFun = opt{LayerNum}.ActFun;
layerType = opt{LayerNum}.type;
dataname = opt{LayerNum}.dataname;
ws = opt{LayerNum}.ws;
num_bases = opt{LayerNum}.num_bases;
pbias = opt{LayerNum}.pbias;
lambda_Dis = opt{LayerNum}.lambda_Dis;
pbias_lambda = opt{LayerNum}.pbias_lambda;
spacing = opt{LayerNum}.spacing;
Wepsilon = opt{LayerNum}.Wepsilon;
Biasepsilon = opt{LayerNum}.Biasepsilon;
nneighbors = opt{LayerNum}.nneighbors;

num_cluster = opt{LayerNum}.num_cluster;
% epsilon = opt{LayerNum}.epsilon;
l2reg = opt{LayerNum}.l2reg;
l1reg = opt{LayerNum}.l1reg;
batch_size = opt{LayerNum}.batch_size;
num_trials = opt{LayerNum}.num_trials;
epdecay = opt{LayerNum}.epdecay;
CD_mode = opt{LayerNum}.CD_mode;
sigma_start = opt{LayerNum}.sigma_start;
sigma_stop = opt{LayerNum}.sigma_stop;
K_CD = opt{LayerNum}.K_CD;
% SetNumber = opt{LayerNum}.SetNumber;
% NumData = opt{LayerNum}.NumData;
LayerNum = opt{LayerNum}.LayerNum;
initialmomentum  = opt{LayerNum}.initialmomentum;
finalmomentum    = opt{LayerNum}.finalmomentum;
DataSetType = opt{LayerNum}.datasetType;
classrbm = opt{LayerNum}.classrbm;
Dis = opt{LayerNum}.Dis;
eta_sparsity = opt{LayerNum}.eta_sparsity;
beta_Dis = opt{LayerNum}.beta_Dis;
MinibatchSize = opt{LayerNum}.MinibatchSize;
RBM_Type = opt{LayerNum}.RBM_Type;
% if mod(ws,2)~=0, error('ws must be even number'); end

bias_mode = 'simple';

% Initialization
W = [];
vbias_vec = [];
hbias_vec = [];
U = [];
lbias_vec = [];
pars = [];

C_sigm = 1;
totalElapsedTime = 0;
% learning
if LayerNum == 1
    numchannels = size(DataSet,3);
else
    numchannels = size(Params{LayerNum-1}.W,4);
end
% Weights Updating mechanism
momentums = [0.5,0.4,0.3,0.2,0.1];
momentumsbias = [0.5,0.4,0.3,0.2,0.1,0];

% Initialize variables
if ~exist('pars', 'var') || isempty(pars)
    pars=[];
end

if ~isfield(pars, 'LayerNum'), pars.LayerNum = LayerNum; end
% if ~isfield(pars, 'SetNumber'), pars.SetNumber = SetNumber; end
if ~isfield(pars, 'DataSetType'), pars.DataSetType = DataSetType; end
if ~isfield(pars, 'classrbm'), pars.classrbm = classrbm; end
if~isfield(pars,'BN'),  pars.BN = BN;   end
if~isfield(pars,'res'),  pars.res = res;   end
if ~isfield(pars, 'ActFun'), pars.ActFun = ActFun; end
if ~isfield(pars, 'Dis'), pars.Dis = Dis; end
if ~isfield(pars, 'RBM_Type'), pars.RBM_Type = RBM_Type; end


if ~isfield(pars, 'num_cluster'), pars.num_cluster = num_cluster; end


if ~isfield(pars, 'ws'), pars.ws = ws; end
if ~isfield(pars, 'num_bases'), pars.num_bases = num_bases; end
if ~isfield(pars, 'nneighbors'), pars.nneighbors = nneighbors; end

if ~isfield(pars, 'spacing'), pars.spacing = spacing; end

if ~isfield(pars, 'pbias'), pars.pbias = pbias; end
if ~isfield(pars, 'lambda_Dis'), pars.lambda_Dis = lambda_Dis; end
if ~isfield(pars, 'beta_Dis'), pars.beta_Dis = beta_Dis; end

if ~isfield(pars, 'pbias_lambda'), pars.pbias_lambda = pbias_lambda; end
if ~isfield(pars, 'bias_mode'), pars.bias_mode = bias_mode; end
if ~isfield(pars, 'eta_sparsity'), pars.eta_sparsity = eta_sparsity; end
if ~isfield(pars, 'Wepsilon'), pars.Wepsilon = Wepsilon; end
if ~isfield(pars, 'Biasepsilon'), pars.Biasepsilon = Biasepsilon; end

% if ~isfield(pars, 'epsilon'), pars.epsilon = epsilon; end
if ~isfield(pars, 'epdecay'), pars.epdecay = epdecay; end
if ~isfield(pars, 'l2reg'), pars.l2reg = l2reg; end
if ~isfield(pars, 'l1reg'), pars.l1reg = l1reg; end
if ~isfield(pars, 'std_gaussian'), pars.std_gaussian = sigma_start; end
if ~isfield(pars, 'sigma_start'), pars.sigma_start = sigma_start; end
if ~isfield(pars, 'sigma_stop'), pars.sigma_stop = sigma_stop; end
if ~isfield(pars, 'layerType'), pars.layerType = layerType; end

if ~isfield(pars, 'K_CD'), pars.K_CD = K_CD; end
if ~isfield(pars, 'CD_mode'), pars.CD_mode = CD_mode; end
if ~isfield(pars, 'C_sigm'), pars.C_sigm = C_sigm; end

if ~isfield(pars, 'num_trials'), pars.num_trials = num_trials; end
% if ~isfield(pars, 'NumData'), pars.NumData = NumData; end
if ~isfield(pars, 'numchannels'), pars.numchannels = numchannels; end
pars.lambda = pars.eta_sparsity;

disp(pars)
if (classrbm == 1) && (Dis == 1)
    error('cannot be true at the same time')
end

%% Initialize weight matrix, vbias_vec, hbias_vec (unless given)

rng('shuffle')
if ~exist('W', 'var') || isempty(W)
    W = 1e-2 *randn(pars.ws,pars.ws, pars.numchannels, pars.num_bases);%./(numchannels);%0.01 normrnd(0,0.01,[pars.ws^2, numchannels, pars.num_bases,batch_number]);%
end
% if LayerNum >5
%     W = 0.01 * W./numchannels;
% end
if ~exist('vbias_vec', 'var') || isempty(vbias_vec)
    vbias_vec = zeros(1,pars.numchannels);
end

if ~exist('hbias_vec', 'var') || isempty(hbias_vec)
    hbias_vec = -0.1.* ones(1,pars.num_bases);%-0.01*ones(pars.num_bases,1);
end

% Initialize variables
if ~exist('pars', 'var') || isempty(pars)
    pars=[];
end
if ~exist('initgamma', 'var') || isempty(pars)
    initgamma = ones(1,num_bases);%0.5.*
end
if ~exist('initbeta', 'var') || isempty(pars)
    initbeta = -0.0* ones(1,num_bases);
end

error_history = [];
sparsity_history = [];

Winc=zeros(size(W));
hbiasinc=zeros(size(hbias_vec));
vbiasinc = zeros(size(vbias_vec));
lbiasinc = 0;
Uinc =0;
Gammainc = 0;
Betainc = 0;


%% Label Parameter inits
rng(1)

if ~exist('U', 'var') || isempty(U)
    U = [];
    pars.numClasses = 1;%size (DataSet.YTrain_onehot,2);
    % %     M = max(pars.numClasses,pars.num_bases );
    % %     max_val = M^(-0.5);
    % %     min_val = - max_val;
    % %     U = min_val + (max_val-min_val).*rand(pars.num_bases,pars.numClasses);
end
if ~exist('lbias_vec', 'var') || isempty(lbias_vec)
    lbias_vec = [];
    %     lbias_vec = zeros(pars.numClasses,1);
end
% load Params_3L_class_SUB1_1

ErrorHistory.AvgEpochTrainError = [];
ErrorHistory.AvgEpochValidError = [];
ErrorHistory.EpochTrainError = [];
ErrorHistory.EpochValidError = [];
ErrorHistory.acc = [];
%
PAR.Winc=zeros(size(W));
PAR.Sdw = 0;
PAR.hbiasinc=zeros(size(hbias_vec));
PAR.Sdhbias = 0;
PAR.vbiasinc = zeros(size(vbias_vec));
PAR.Sdvis = 0;
PAR.Gammainc = 0;
PAR.Betainc = 0;
PAR.SdG = 0;
PAR.SdB = 0;
PAR.t = 0;

%%

s = size(W);
% num_clusters = 2;
CRBM.Cluster_centers = randn(pars.num_cluster,pars.num_bases);
CRBM.W1 = 1e-2 *randn(1,1, s(end-1), s(end));
CRBM.W00 = 1e-2 *randn(3,3, pars.num_bases, pars.num_bases);
CRBM.W01 = 1e-2 *randn(1,1, pars.num_bases, 1);
CRBM.W = W;%Params{1}.W;
CRBM.hbias_vec = hbias_vec;
CRBM.vbias_vec = vbias_vec;
CRBM.U = U;
CRBM.lbias_vec = lbias_vec;
CRBM.Gamma = initgamma;
CRBM.Beta = initbeta;

PAR.W00inc = zeros(size(CRBM.W00));
PAR.S00dw = 0;
PAR.W01inc = zeros(size(CRBM.W01));
PAR.S01dw = 0;
% Cluster weights and bias
PAR.ckinc = zeros(size(CRBM.Cluster_centers));
PAR.Sdck = 0;
PAR.dck = 0;

CRBM.Hiddensparsity = pars.pbias;
CRBM.classrbm = classrbm;
CRBM.Dis = Dis;
CRBM.runningavg_prob = [];
CRBM.runningavg_exp = [];

CRBM.posmuk = 0;
CRBM.posvark = 0;
% CRBM.negmuk = 0;
% CRBM.negvark = 0;


%%
NumData = min(1000,size(DataSet.XTrain,4));%size(DataSet,4);%
batchArraySize = ceil(NumData/MinibatchSize);
% if LayerNum == 1
%     DataSet =  DataSet;%sqrt(0.1) .*
% else
%     DataSet =  DataSet;%(sqrt(0.1)^-1) .*
% end

a = mean(DataSet.XTrain,4);
b = std(DataSet.XTrain,[],4);
%         DataSet.XTrain = bsxfun(@minus,DataSet.XTrain,a);
% DataSet.XTrain = bsxfun(@rdivide,bsxfun(@minus,DataSet.XTrain,a),1);
rng('shuffle');
imidx_batch = randsample(size(DataSet.XTrain,4),size(DataSet.XTrain,4));


%%
% fprintf('********  Learning strats  ***********\n');
t = 0;
k = 0;
Kurtwm = zeros(pars.num_trials+1,1);%size(CRBM.W,3));
negentwm = zeros(pars.num_trials+1,1);%size(CRBM.W,3));
[Kurtwm(1,:),negentwm(1,:)] = NON_Gaussianity_measur(CRBM.W);
pars.nClasses = 1;%length(unique( DataSet.YTrain));
% CRBM.hMaps_initi = ones(1,pars.num_bases);
CRBM.hMaps = [];
PAR.current_epsilonW = pars.Wepsilon;
PAR.current_epsilonB = pars.Biasepsilon;
PAR.momentum = initialmomentum ;
PAR.momentumf = finalmomentum;
acc = [];

avgstart = pars.num_trials + t - 0.1*pars.num_trials;
t0 = 0; % a controling key to use all the images
pars.count = 0;
pars.nf = 0;
pars.nBN = 0;
for t=1:pars.num_trials
    pars.t = t;
    pars.k = k;
    %%
    if (pars.t == 20 && (pars.num_cluster > 0))% initialization of cluster centers
        Fs = [];
        n =400; 
        nts = size(DataSet.XTrain,4);
        nts = randi(nts,1,n);
        for j = 1:n
            im = DataSet.XTrain(:,:,:,nts(j));
            %         im = standard_im(im);
            [ Fs(j,:,:,:) ] = FeatExtct_1( im,Params,LayerNum );
        end
        X = reshape(Fs,[size(nts,2)*size(Fs,2)*size(Fs,2),size(Fs,4)]);

        opts = statset('Display','final','UseParallel',1);
        [idx,C] = kmeans(X,pars.num_cluster,'Distance','sqeuclidean',...
            'Replicates',5,'Options',opts);
        CRBM.Cluster_centers = C;
        clear Fs X im opts idx C
        delete(gcp('nocreate'))
    end
    %%
    % weight decay
    % Take a random permutation of the samples
    %     epsilon = pars.epsilon/(1+epdecay*t);
    %         PAR.current_epsilonW = 0.95^pars.t * pars.Wepsilon;
    %         PAR.current_epsilonB = 0.95^pars.t * pars.Biasepsilon;

    if PAR.current_epsilonW > 1e-6
        PAR.current_epsilonW = pars.Wepsilon/(1+epdecay*(t));
        %         PAR.current_epsilonW = pars.Wepsilon * exp(-0.1*t);
    end
    if PAR.current_epsilonB > 1e-6
        PAR.current_epsilonB = pars.Biasepsilon/(1+epdecay*(t));
        %         PAR.current_epsilonB = pars.Biasepsilon * exp(-0.1*t);
    end

    t0 = t0 + 1;
    if t0 <= ceil(size(DataSet.XTrain,4)/NumData)
        Subimidx_batch = imidx_batch((t0-1)*NumData+1:min(t0*NumData,end));
        batchArraySize = ceil(length(Subimidx_batch)/MinibatchSize);
        if t0 == ceil(size(DataSet.XTrain,4)/NumData)
            t0 = 0;
            %             rng('shuffle');
            %             imidx_batch = randsample(size(DataSet.XTrain,4),size(DataSet.XTrain,4));
        end
    end
    t_time = tic;
    ferr_current_iter = 0;
    sparsity_curr_iter = [];
    loss_curr_iter = [];
    for batchNumber = 1:batchArraySize

        imidx = Subimidx_batch((batchNumber-1)*MinibatchSize+1:min(batchNumber*MinibatchSize,end));
        imdata = DataSet.XTrain(:,:,:,imidx);
        train_y = [];%DataSet.YTrain_onehot(imidx,:);
        pars.count = pars.count + 1;
        temp =  imdata;%(0.66 / sqrt(0.1) ).*
        %         if LayerNum > 1
        %             temp =    0.66.* imdata;%standardize_im(imdata);
        %         else
        %             temp =    sqrt(0.1)* imdata;%standardize_im(imdata);
        %         end
        %         temp = temp - mean(temp,'all');
%         if rand()> 0.5
%             temp = fliplr(temp);
%         end
        if LayerNum > 1
            for j = 1 : LayerNum - 1
                [Epool,~,~] = netOutLayer2(temp,Params{j});
                %                 [dim1] = find(outmap>0);
                %                 Hunpool = convs(temp,Params{j}.CRBM.W0 , 0);
                %                 [Hpool,~] = convnet_maxpool(Hunpool,Params{j}.pars.spacing);
                %                 Hpool = reshape(Hunpool(dim1),size(Epool));
                %                 Epool = 0.5*(Hpool+sigmoid(Epool));%sigmoid(Epool+Hpool);
                if  strcmp(opt{j+1}.type,'g')
                    temp =    standard_im(Epool);
                elseif strcmp(opt{j+1}.type,'b')
                    temp = Epool;
                end
            end
        end


        %         if LayerNum == 1
        %             batch_ws = [70,70];
        %             imdata_batch = trim_image_square(temp,ws,batch_ws,spacing);
        %         else
        batch_ws = [size(temp,1),size(temp,2)];
        imdata_batch = trim_image_square(temp,ws,batch_ws,spacing);
        %         end
        if strcmp(layerType,'g')
            imdata_batch = (imdata_batch - mean(imdata_batch(:)));
        end
        padsize = floor(((size(CRBM.W,1)) - 1) / 2);
        imdata_batch = padarray(imdata_batch,[padsize,padsize],'replicate');
        pars.Wim = size(imdata_batch,2);
        pars.Him = size(imdata_batch,1);
        pars.Hhidden = pars.Him - pars.ws +1;
        pars.Whidden = pars.Wim - pars.ws +1;
        if (mean(CRBM.hMaps,"all")==0 || isempty(CRBM.hMaps))
            CRBM.hMaps = zeros([pars.Hhidden,pars.Whidden,2]);%pars.num_bases]);
            CRBM.hMaps_init = 1;
            % CRBM.hMaps  = trim_image_for_spacing(CRBM.hMaps, ws, spacing);
            S = size(CRBM.hMaps);
            S(3) = 1;
            CRBM.runningavg_exp = zeros([S(1),S(2),pars.num_bases,S(3)]);% ?
        end
        % update rbm
        [ferr, PAR, poshidprobs, ~, ~,CRBM ]=...
            DCRBM_GEN(imdata_batch, train_y,CRBM, pars,PAR,...
            bias_mode, l2reg,l1reg);
        ferr_current_iter = [ferr_current_iter, ferr.ferr_neg];
        sparsity_curr_iter = [sparsity_curr_iter, mean(poshidprobs(:))];
        if  pars.t > 0 && pars.Dis
            pars.nf = pars.nf + 1;
            [PAR,CRBM] = Hmaps_estimation (CRBM,pars,PAR);
        end
        %         PAR.GradW = 0;
        %         PAR.Gradb = 0;

        if  (pars.t > 19 && (pars.num_cluster > 0))
            [PAR,euclid_dis,soft_x] = dis_gradient_forward (imdata_batch ,CRBM,PAR);
            loss_cluster = mean( euclid_dis.*soft_x ,[1,2,3]);
            %             loss_cluster_map = mean( euclid_dis.*soft_x ,[1,2,3]);
            loss = 0.1*loss_cluster;
            loss_curr_iter = [loss_curr_iter, loss];



            %             [distances, soft_assign] = kmeans_deep(maps,PAR,CRBM);
            %             [PAR,CRBM,maps] = dis_gradient_backWard (poshidprobs,CRBM,pars,PAR);

        else
            PAR.GradW = 0;
            PAR.Gradb = 0;
        end
        PAR = GradientSelection (PAR,CRBM,pars);
       if pars.count < 5000
           pars.pbias_lambda = pars.count/5000 * 5 + (1 - pars.count/5000) * pars.pbias_lambda;
%             PAR.momentum = pars.count/5000 * PAR.momentumf + (1 - pars.count/5000) * PAR.momentum;
        else
%             PAR.momentum = PAR.momentumf;
            pars.pbias_lambda = 5;
        end
        if pars.t <5
            PAR.Wmomentum = initialmomentum;% PAR.momentum;%momentums(min(pars.t,end));
            PAR.Biasmomentum = initialmomentum;%momentumsbias(min(pars.t,end));
        else
            PAR.Wmomentum = finalmomentum ;% PAR.momentum;%momentums(min(pars.t,end));
            PAR.Biasmomentum = finalmomentum;%momentumsbias(min(pars.t,end));
        end
        % update parameters
        [CRBM,PAR,pars] = ApplyGrads(CRBM,PAR,pars,avgstart,layerType);
    end  %% END of Epoch
    tElapsed = toc(t_time);
    if (pars.std_gaussian > pars.sigma_stop) % stop decaying after some point
        pars.std_gaussian = pars.std_gaussian*0.99;
    end
    [Kurtwm(t+1,:),negentwm(t+1,:)] = NON_Gaussianity_measur(CRBM.W);
    totalElapsedTime = totalElapsedTime + tElapsed;
    time_history(1,t) = tElapsed;
    error_history(1,t) = mean(ferr_current_iter);
    sparsity_history(1,t) = mean(sparsity_curr_iter);
    loss_history(1,t) = mean(loss_curr_iter);
    if 0
        if LayerNum == 1
            figure(1);display_network_nonsquare(CRBM.W);
            title(sprintf('Layer %d Epoch %d %s',LayerNum,t,opt{LayerNum}.dataname)),drawnow
            saveas(gcf, sprintf('../Bases Visualization/%s/Layer%d_%s_%04d.png',dataname, LayerNum,'bases', t));
        elseif LayerNum == 2
            figure(2);display_crbm_v2_bases(CRBM.W, Params{1}, Params{1}.pars.spacing);
            title(sprintf('Layer %d Epoch %d %s',LayerNum,t, opt{LayerNum}.dataname)),drawnow
            saveas(gcf, sprintf('../Bases Visualization/%s/Layer%d_%s_%04d.png',dataname, LayerNum,'bases', t));
        elseif LayerNum == 3
            figure(3);V1.W = display_crbm_v3_bases(CRBM.W, Params{2}, Params{2}.pars.spacing);
            title(sprintf('Layer %d Epoch %d %s',LayerNum,t, opt{LayerNum}.dataname)),drawnow
            figure(4);display_crbm_v2_bases(V1.W, Params{1}, Params{1}.pars.spacing);
            title(sprintf('Layer %d Epoch %d %s',LayerNum,t, opt{LayerNum}.dataname)),drawnow
            saveas(gcf, sprintf('../Bases Visualization/%s/Layer%d_%s_%04d.png',dataname, LayerNum,'bases', t));
        end
    end
    params.pars = pars;
    params.W = CRBM.W;
    params.vbias_vec = CRBM.vbias_vec;
    params.hbias_vec = CRBM.hbias_vec;
    params.U = CRBM.U;
    params.lbias_vec = CRBM.lbias_vec;
    params.CRBM = CRBM;
    Params{LayerNum} = params;
    clear params
    %     if mod(t,15) == 0
    %         T_snePlot( Params,DataSet.XTrain,DataSet.YTrain,opt)
    %     end
    %%
    if ~isequal(lower(dataname),'olshausen')
        if 0%  t<5 ||(mod(t,5) == 0)
            acc1 = acc_Epoch(CRBM, pars,dataname,LayerNum,Params,SetNumber,DataSet);
            acc.test(t) = acc1.testACC;
            acc.train(t) = acc1.trainAcc;
            acc.AUCtest(t) = acc1.AUCtest;
            acc.AUCtrain(t) = acc1.AUCtrain;
            figure(0101),plot(acc.test)
            hold on; plot(acc.train), hold off
            figure(0111),plot(acc.AUCtest)
            hold on,plot(acc.AUCtrain), hold off
            fprintf('ACC_test = %f\t ACC_train = %f \nAUC_test = %f\t AUC_train = %f \n',acc.test(t),acc.train(t),acc.AUCtest(t),acc.AUCtrain(t));
        end
    end
    %         figure(19),plot(Kurtwm(1:t+1,:)),title('overal Excess Kurtosis');
    %     figure(20),plot(negentwm(1:t+1,:));title('overal negentropy')

    fprintf('epoch %3d error = %4.5g \tsparsity_hid = %4.5g \t||dW|| = %4.5g \tcluster_loss = %4.5g',...
        t, mean(ferr_current_iter), mean(sparsity_curr_iter),sqrt(sum(PAR.dW(:).^2)), mean(loss_curr_iter));
    fprintf('\tElapsed time =  %4.5g \n', tElapsed);

end
params.CRBM = CRBM;
params.pars = pars;
params.W = CRBM.W;
params.vbias_vec = CRBM.vbias_vec;
params.hbias_vec = CRBM.hbias_vec;
params.U = CRBM.U;
params.lbias_vec = CRBM.lbias_vec;
params.Gamma = CRBM.Gamma;
params.Beta = CRBM.Beta;
params.posmuk = CRBM.posmuk;
params.posvark = CRBM.posvark;
% params.negmuk = CRBM.negmuk;
% params.negvark = CRBM.negvark;
params.error_history = error_history;
params.sparsity_history = sparsity_history;
params.Kurtwm = Kurtwm;
params.negentwm = negentwm;
params.time_history = time_history;
params.loss_history = loss_history;
end
