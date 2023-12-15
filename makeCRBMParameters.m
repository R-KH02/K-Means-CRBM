function Opt = makeCRBMParameters(numofLayers,MinibatchSize,dataname)

    L1.num_cluster = 10;
    L2.num_cluster = 8;
    L3.num_cluster = 8;
    L4.num_cluster = 8;
    L5.num_cluster = 8;
    L6.num_cluster = 32;

    
    L1.LayerNum = 1;
    L2.LayerNum = 2;
    L3.LayerNum = 3;
    L4.LayerNum = 4;
    L5.LayerNum = 5;
    L6.LayerNum = 6;
    % Layer type
    L1.type = 'g';
    L2.type = 'b';
    L3.type = 'b';
    L4.type = 'b';
    L5.type = 'b';
    L6.type = 'b';
    % Data name
    L1.dataname = dataname;%'CFAR10';%Olshausen caltech
    L2.dataname = dataname;%'MacMaster';caltech
    L3.dataname = dataname;%'MacMaster';
    L4.dataname = dataname;%'MacMaster';
    L5.dataname = dataname;%'MacMaster';
    L6.dataname = dataname;%'MacMaster';
    % Data Typ
    L1.datasetType = 'Natural images';
    L2.datasetType = 'warpped';%'fullsized';%
    L3.datasetType = 'warpped';
    L4.datasetType = 'warpped';
    L5.datasetType = 'warpped';
    L6.datasetType = 'warpped';
    % Filter Size  based on input size of 116*116
    L1.ws = 7; %
    L2.ws = 5; %
    L3.ws = 5; %
    L4.ws = 5; %
    L5.ws = 3; %
    L6.ws = 3; %
    % res
        % res
    L1.res = 0; %
    L2.res = 0; %
    L3.res = 0; %
    L4.res = 0; %
    L5.res = 0; %
    L6.res = 0; %

    % Bases number
    L1.num_bases = 32;
    L2.num_bases = 40;
    L3.num_bases = 60;
    L4.num_bases = 60;
    L5.num_bases = 128;
    L6.num_bases = 256;
    % Batch Normalization
    L1.BN = 0;
    L2.BN = 0;
    L3.BN = 0;
    L4.BN = 0;
    % Sparsity Value
    L1.pbias = 0.005; %changed from 0.001618 to spacing 2
    L2.pbias = 0.005;
    L3.pbias = 0.005;
    L4.pbias = 0.05;
    L5.pbias = 0.05;
    L6.pbias = 0.05;
    %% Dis lambda
    L1.lambda_Dis = 1e-2;
    L2.lambda_Dis = 1e-2;%0.005;
    L3.lambda_Dis = 1e-2;
    L4.lambda_Dis = 1e-2;
    L5.lambda_Dis = 1e-3;
    L6.lambda_Dis = 1;
    %% Beta Dis
    L1.beta_Dis = 1e3;
    L2.beta_Dis = 1e3;
    L3.beta_Dis = 1e3;
    L4.beta_Dis = 1e3;
    L5.beta_Dis = 3;
    L6.beta_Dis = 3;
    
    %% Activation function probmax max sigm
    L1.ActFun = 'sigm';
    L2.ActFun = 'sigm';
    L3.ActFun = 'sigm';
    L4.ActFun = 'sigm';
    L5.ActFun = 'sigm';
    L6.ActFun = 'sigm';
    % Sparsity Gain
    L1.pbias_lambda = 5; 
    L2.pbias_lambda = 5;
    L3.pbias_lambda = 5;
    L4.pbias_lambda = 5;
    L5.pbias_lambda = 5;
    L6.pbias_lambda = 0.1;
    % ets_sparsity
    L1.eta_sparsity = 0.01;
    L2.eta_sparsity = 0.01;
    L3.eta_sparsity = 0.01;
    L4.eta_sparsity = 0.01;
    L5.eta_sparsity = 0.01;
    L6.eta_sparsity = 0.01;
    % Spacing
    L1.spacing = 2;
    L2.spacing = 2;
    L3.spacing = 2;
    L4.spacing = 2;
    L5.spacing = 2;
    L6.spacing = 1;
    % Learning Rate
    
    L1.Wepsilon = 1e-2;
    L1.Biasepsilon = 1e-2;
    L2.Wepsilon = 1e-2;
    L2.Biasepsilon = 1e-2;
    L3.Wepsilon = 1e-2;
    L3.Biasepsilon = 1e-2;
    L4.Wepsilon = 1e-2;
    L4.Biasepsilon = 1e-2;
    L5.Wepsilon = 1e-3;
    L5.Biasepsilon = 1e-2;
    L6.Wepsilon = 1e-3;
    L6.Biasepsilon = 1e-3;
    L7.Wepsilon = 1e-3;
    L7.Biasepsilon = 1e-3;
    
    L1.nneighbors = floor(0.75 * L1.num_bases);
    L2.nneighbors = floor(1 * L2.num_bases);
    L3.nneighbors = floor(1 * L3.num_bases);
    L4.nneighbors = floor(1 * L4.num_bases);
    L5.nneighbors = floor(1 * L5.num_bases);
    L6.nneighbors = floor(1 * L6.num_bases);
    
    L1.epsilon = 0.01;
    L2.epsilon = 0.01;
    L3.epsilon = 0.01;
    L4.epsilon = 0.01;
    L5.epsilon = 0.01;
    L6.epsilon = 0.01;
    % eps decay value
    L1.epdecay = 0.01;%1e-4;
    L2.epdecay = 0.01;
    L3.epdecay = 0.01;
    L4.epdecay = 0.01;
    L5.epdecay = 0.01;
    L6.epdecay = 0.01;
    % L2 Regularizaion value
    L1.l2reg = 0.0001;
    L2.l2reg = 0.0001;
    L3.l2reg = 0.0001;
    L4.l2reg = 0.0001;
    L5.l2reg = 0.0001;
    L6.l2reg = 0.0001;
    
    % Contrastive divergence mode
    L1.CD_mode = 'exp';% or 'exp'mf;
    L2.CD_mode = 'exp';
    L3.CD_mode = 'exp';
    L4.CD_mode = 'exp';
    L5.CD_mode = 'exp';
    L6.CD_mode = 'exp';
    % CD iteration num
    L1.K_CD = 1;
    L2.K_CD = 1;
    L3.K_CD = 1;
    L4.K_CD = 1;
    L5.K_CD = 1;
    L6.K_CD = 1;
    % Batch size
    L1.batch_size = 1; % set it to 1 in the case of CFAR10 database
    L2.batch_size = 1;
    L3.batch_size = 1;
    L4.batch_size = 1;
    L5.batch_size = 1;
    L6.batch_size = 1;
    % Num of trials
    L1.num_trials = 350;
    L2.num_trials = 100;
    L3.num_trials = 100;
    L4.num_trials = 110;
    L5.num_trials = 100;
    L6.num_trials = 100;
    % Sigma values
    L1.sigma_start = 0.2;
    L2.sigma_start = 0.2;
    L3.sigma_start = 0.2;
    L4.sigma_start = 0.2;
    L5.sigma_start = 0.2;
    L6.sigma_start = 0.2;
    
    if strcmp(L1.type,'g')
        L1.sigma_stop = 0.1;
    elseif strcmp(L1.type,'b')
        L1.sigma_stop = 0.2;
    end
    if strcmp(L2.type,'g')
        L2.sigma_stop = 0.1;
    elseif strcmp(L2.type,'b')
        L2.sigma_stop = 0.2;
    end
    if strcmp(L3.type,'g')
        L3.sigma_stop = 0.1;
    elseif strcmp(L3.type,'b')
        L3.sigma_stop = 0.2;
    end
    if strcmp(L4.type,'g')
        L4.sigma_stop = 0.1;
    elseif strcmp(L4.type,'b')
        L4.sigma_stop = 0.2;
    end
    if strcmp(L5.type,'g')
        L5.sigma_stop = 0.1;
    elseif strcmp(L5.type,'b')
        L5.sigma_stop = 0.2;
    end
    if strcmp(L5.type,'g')
        L6.sigma_stop = 0.1;
    elseif strcmp(L5.type,'b')
        L6.sigma_stop = 0.2;
    end
        
    % Num data for CDBN training phase
    L1.NumData = 100;%200
    L2.NumData = 1000;%200
    L3.NumData = 1000;%100
    L4.NumData = 1000;%100
    L5.NumData = 200;%100
    L6.NumData = 200;%100
    % Momentum values
    L1.initialmomentum  = 0.5;
    L1.finalmomentum    = 0.9;
    L2.initialmomentum  = 0.5;
    L2.finalmomentum    = 0.9;
    L3.initialmomentum  = 0.5;
    L3.finalmomentum    = 0.9;
    L4.initialmomentum  = 0.5;
    L4.finalmomentum    = 0.9;
    L5.initialmomentum  = 0.5;
    L5.finalmomentum    = 0.9;
    L6.initialmomentum  = 0.5;
    L6.finalmomentum    = 0.9;
    %
    % nameofDatasetAdd = DatasetAdd_extraction() ;
    nameofDatasetAdd = 'DataSetAdd_Bin';%'DataSetAdd';
    L1.nameofDatasetAdd = nameofDatasetAdd;
    L2.nameofDatasetAdd = nameofDatasetAdd;
    L3.nameofDatasetAdd = nameofDatasetAdd;
    L4.nameofDatasetAdd = nameofDatasetAdd;
    L5.nameofDatasetAdd = nameofDatasetAdd;
    L6.nameofDatasetAdd = nameofDatasetAdd;
    %
    scl = 1;
    L1.scl = scl;
    L2.scl = scl;
    L3.scl = scl;
    L4.scl = scl;
    L5.scl = scl;
    L6.scl = scl;
    %%
    L1.classrbm = 0;
    L2.classrbm = 0;
    L3.classrbm = 0;
    L4.classrbm = 0;
    L5.classrbm = 0;
    L6.classrbm = 0;
    
    %%
    L1.Dis = 0;
    L2.Dis = 0;
    L3.Dis = 1;
    L4.Dis = 1;
    L5.Dis = 1;
    L6.Dis = 0;
    %
    
    %%
    L1.RBM_Type = 'gen';%generative
    L2.RBM_Type = 'gen';%discriminative
    L3.RBM_Type = 'gen';%
    L4.RBM_Type = 'gen';%
    L5.RBM_Type = 'gen';%
    L6.RBM_Type = 'gen';%
    %%
    L1.alpha = 0.01;
    L2.alpha = 0.01;
    L3.alpha = 0.01;
    L4.alpha = 0.01;
    L5.alpha = 0.01;
    L6.alpha = 0.01;
    %%
    L1.l1reg = 0;
    L2.l1reg = 1e-5;
    L3.l1reg = 3e-5;
    L4.l1reg = 0;
    L5.l1reg = 0;
    L6.l1reg = 0;
    
    L1.MinibatchSize = MinibatchSize;
    L2.MinibatchSize = MinibatchSize;
    L3.MinibatchSize = MinibatchSize;
    L4.MinibatchSize = MinibatchSize;
    L5.MinibatchSize = MinibatchSize;
    L6.MinibatchSize = MinibatchSize;
    
    
    %%
    L1.batch_ws_ratio = 1/3;
    L2.batch_ws_ratio = 1/3;
    L3.batch_ws_ratio = 2/3;
    L4.batch_ws_ratio = 2/3;
    L5.batch_ws_ratio = 2/3;
    L6.batch_ws_ratio = 2/3;
    %%
    Opt{1} = L1;
    Opt{2} = L2;
    Opt{3} = L3;
    Opt{4} = L4;
    Opt{5} = L5;
    Opt{6} = L6;
    clear L1 L2 L3 L4 L5 L6


end
