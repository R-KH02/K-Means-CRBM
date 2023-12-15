function outR = Feature_Extraction(file_name,DataSet,dataname,type)

outR = [];
%% Load input data
Softmax_use = 1;
rndforest_use = 0;
knn_use = 1;
%% Load network parameteres
Xtemp = load(sprintf('../Params/%s',file_name));

Params = Xtemp.params;
% a =  1;%(sqrt(0.1)^-1) ;
a = 0.66/sqrt(0.1);%/sqrt(0.1);%(sqrt(0.1)/1.5);%0.66;
typeofsum = 'cat';
lambda = 1e-2;
%% load network Parameters
NumLayers = size(Params,2);
%% extract features
FeatOfLayer = [1 0 0 0 0 0 0];
SL1_features = [1 1 1 0 0 1 1];
testing_FE = 1;
training_FE = 1;
if training_FE
%     if isempty(gcp('nocreate'))
%         pool = parpool;
%     end
switch type
    case 'test'
        SVHNData.XTrain = a*cat(4,DataSet.XTrain,DataSet.XValidation);%permute(dataset_n.XTrain(:,:,1:2000),[1,2,4,3]);% sqrt(0.1)
        SVHNData.YTrain = cat(2,DataSet.YTrain,DataSet.YValidation);
        SVHNData.XTest = a*DataSet.XTest;%permute(dataset_n.XTrain(:,:,end-999:end),[1,2,4,3]);
        SVHNData.YTest = DataSet.YTest;
    case 'valid_mode'
        SVHNData.XTrain = a * DataSet.XTrain;%permute(dataset_n.XTrain(:,:,1:2000),[1,2,4,3]);% sqrt(0.1)
        SVHNData.YTrain = DataSet.YTrain;
        SVHNData.XTest = a * DataSet.XValidation;
        SVHNData.YTest = DataSet.YValidation;
end
    a = mean(SVHNData.XTrain,4);
    b = 1;%std(SVHNData.XTrain,[],4);
    
    % SVHNData.XTrain = bsxfun(@minus,SVHNData.XTrain,a);%
    % SVHNData.XTest = bsxfun(@minus,SVHNData.XTest,a);%
    SVHNData.XTrain = bsxfun(@rdivide,bsxfun(@minus,SVHNData.XTrain,a),b);
    SVHNData.XTest = bsxfun(@rdivide,bsxfun(@minus,SVHNData.XTest,a),b);
    
    
    clear DataSet
    
    % im = SVHNData.XTrain(:,:,:,1);
    % [ Fs ] = FeatExtct( im,Params,NumLayers,SL1_features );
    Feat_train = cell(1,NumLayers);
    Feat_test = cell(1,NumLayers);
    
    fprintf('\n*******\tTraining features Extracting ...\t ***************\n')
    % Extract features from training data
    XTrain = SVHNData.XTrain;
    nts = size(XTrain,4);
    parfor j = 1:nts
        im = XTrain(:,:,:,j);
%         im = standard_im(im);
        [ Fs{j} ] = FeatExtct( im,Params,NumLayers,SL1_features );
    end
    for i2 = 1:NumLayers
        if SL1_features (i2)
            xtemp = zeros([size(Fs{1}{i2}),size(Fs,2)]);
            for j = 1:nts
                xtemp(:,:,:,j) = Fs{1,j}{1,i2};
            end
            Feat_train{1,i2} = xtemp;
        end
    end
    clear Fs
    if ~exist(sprintf('../Data_L'),'dir')
        mkdir('../Data_L')
    end
    for i = 1:1
        for i2 = 1 : NumLayers
            if SL1_features(i2)
                f_name = sprintf('train_data_Layer%d_%s.mat',i2,dataname);
                temp = Feat_train{1,i2};
                save (sprintf('../Data_L/%s',f_name),'temp','-v7.3')
                clear temp
            end
        end
    end
    clear Feat_train
    SVHNData.XTrain = [];
end
clear Feats
if testing_FE
    fprintf('\n*******\tTest features Extracting ...\t ***************\n')
    % Extract features from testing data
    nts = size(SVHNData.XTest,4);
    XTest = SVHNData.XTest;
    parfor i = 1:nts
        im = XTest(:,:,:,i);
%         im = standard_im(im);
        [ Fs{i} ] = FeatExtct( im,Params,NumLayers,SL1_features );
    end
    for i2 = 1:NumLayers
        if SL1_features (i2)
            xtemp = zeros([size(Fs{1}{i2}),size(Fs,2)]);
            for j = 1:nts
                xtemp(:,:,:,j) = Fs{1,j}{1,i2};
            end
            Feat_test{1,i2} = xtemp;
        end
    end
    clear Fs
    for i = 1:1
        for i2 = 1 : NumLayers
            if SL1_features(i2)
                f_name = sprintf('test_data_Layer%d_%s.mat',i2,dataname);
                temp = Feat_test{i,i2};
                save (sprintf('../Data_L/%s',f_name),'temp','-v7.3')
                clear temp
            end
        end
    end
    
    clear Feat_test  Feats
    SVHNData.XTest= [];
    
    
    train_label = SVHNData.YTrain;
    save (sprintf('../Data_L/train_Label_%s.mat',dataname), 'train_label','-v7.3')
    test_label = SVHNData.YTest;
    save (sprintf('../Data_L/test_label_%s.mat',dataname), 'test_label','-v7.3')
end % if 0
%% Train A classifier

Train_Features = [];
Test_Features = [];
ltemp = load (sprintf('../Data_L/train_Label_%s.mat',dataname), 'train_label');
train_label = ltemp.train_label;
ltemp = load (sprintf('../Data_L/test_label_%s.mat',dataname), 'test_label');
test_label = ltemp.test_label;
clear ltemp
for i2 = 1 : NumLayers
    if FeatOfLayer(i2)
        f_name = sprintf('train_data_Layer%d_%s.mat',i2,dataname);
        
        load (sprintf('../Data_L/%s',f_name),'temp')
        temp = reshape(temp,[],size(temp,4));
        Feat_train{1,i2} = temp;
        clear temp
    end
end
feats_train = [];
for j = 1:NumLayers
    if FeatOfLayer(j)
        switch typeofsum
            case 'cat'
                temp1 = Feat_train{1,j};
            case 'sum'
                t1 = size(Feat_train{1,j},2) / size(Params{1,j}.W,3);
                temp1 = sum(reshape(Feat_train{1,j},[size(Feat_train{1,j},1),t1,size(Params{1,j}.W,3)]),3);
            case 'mean'
                t1 = size(Feat_train{1,j},2) / size(Params{1,j}.W,3);
                temp1 = mean(reshape(Feat_train{1,j},[size(Feat_train{1,j},1),t1,size(Params{1,j}.W,3)]),3);
            case 'meanmax'
                t1 = size(Feat_train{1,j},2) / size(Params{1,j}.W,3);
                temp21 = mean(reshape(Feat_train{1,j},[size(Feat_train{1,j},1),t1,size(Params{1,j}.W,3)]),3);
                temp2 = max(reshape(Feat_train{1,j},[size(Feat_train{1,j},1),t1,size(Params{1,j}.W,3)]),[],3);
                temp1 = sqrt(abs(temp21.*temp2));
        end
        feats_train = cat(1,feats_train,temp1);
    end
end
Train_Features = feats_train;    clear feats_train
%% training a Binary SVM
%% load test features
for i2 = 1 : NumLayers
    if FeatOfLayer(i2)
        f_name = sprintf('test_data_Layer%d_%s.mat',i2,dataname);
        
        load (sprintf('../Data_L/%s',f_name),'temp')
        temp = reshape(temp,[],size(temp,4));
        Feat_test{1,i2} = temp;
        clear temp
    end
end

feats_test = [];
for j = 1:NumLayers
    if FeatOfLayer(j)
        switch typeofsum
            case 'cat'
                temp1 = Feat_test{1,j};
            case 'sum'
                t1 = size(Feat_test{1,j},2) / size(Params{1,j}.W,3);
                temp1 = sum(reshape(Feat_test{1,j},[size(Feat_test{1,j},1),t1,size(Params{1,j}.W,3)]),3);
            case 'mean'
                t1 = size(Feat_test{1,j},2) / size(Params{1,j}.W,3);
                temp1 = mean(reshape(Feat_test{1,j},[size(Feat_test{1,j},1),t1,size(Params{1,j}.W,3)]),3);
            case 'meanmax'
                t1 = size(Feat_test{1,j},2) / size(Params{1,j}.W,3);
                temp21 = mean(reshape(Feat_test{1,j},[size(Feat_test{1,j},1),t1,size(Params{1,j}.W,3)]),3);
                temp2 = max(reshape(Feat_test{1,j},[size(Feat_test{1,j},1),t1,size(Params{1,j}.W,3)]),[],3);
                temp1 = sqrt(abs(temp21.*temp2));
        end
        feats_test = cat(1,feats_test,temp1);
    end
end


Test_Features = feats_test;    clear feats_test


clear SVHNData PainDataSet
delete(gcp('nocreate'))
%%
Features.trainingLabels = train_label; clear train_label
Features.trainingFeatures = Train_Features; clear Train_Features
Features.testFeatures = Test_Features;clear Test_Features
Features.testLabels = test_label; clear test_label


%%
if Softmax_use
    %     Features.trainingFeatures = reshape(Features.trainingFeatures,[
    % set learning parameters
    numClasses = length(unique(Features.trainingLabels));     % Number of classes
    %     lambda = 1e-6; % Weight decay parameter
    % load taining data and get the input size
    inputSize = size(Features.testFeatures,1);
    if iscategorical( Features.trainingLabels)
        trainingLabels = converttoNumbers(Features.trainingLabels);%converttoNumbers
    else
        trainingLabels = Features.trainingLabels ;
        trainingLabels = trainingLabels;
    end
    %%======================================================================
    %% STEP 4: Learning parameters
    a1=size(Features.trainingFeatures,2);a2 = size(Features.trainingFeatures,1);
    P = find(FeatOfLayer == 1);
    fmt = ['Layers: [', repmat('%g, ', 1, numel(P)-1), '%g] features with # of samples X # of features : %d  X %d\n'];
    fprintf(fmt ,P,a1,a2)
    options.maxIter = 10000;
    softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
        Features.trainingFeatures, trainingLabels, options);
    %% STEP 5: Testing
    if iscategorical(Features.testLabels)
        testLabels = converttoNumbers(Features.testLabels);%converttoNumbers
    else
        testLabels = Features.testLabels;
        testLabels = testLabels;
    end
    % You will have to implement softmaxPredict in softmaxPredict.m
    [pred,score] = softmaxPredict(softmaxModel, Features.testFeatures);
    auc = multiClassAUC(score', testLabels);
    %      [~,~,~,AUCpr] = perfcurve(testLabels, score(1,:), 1, 'xCrit', 'reca', 'yCrit', 'prec')
    acc.SM = mean(testLabels(:) == pred(:));
    %     plotconfusion(oneHot(testLabels,numClasses,unique(trainingLabels))',oneHot(pred,numClasses,unique(trainingLabels))');
        cm = confusionchart(testLabels,pred);
    cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'Confusion Matrix';
outR.cmSM = cm;
outR.auc = auc;
outR.SM_ACC = acc.SM;
    outR.testLabels = testLabels;
    outR.pred =pred;
    outR.score = score;

%         net = trainSoftmaxLayer(Features.trainingFeatures,oneHot(Features.trainingLabels,numClasses,unique(Features.trainingLabels))','MaxEpochs',2000);
% %     deepnet1 = train(net,Features.trainingFeatures,oneHot(Features.trainingLabels,numClasses,unique(Features.trainingLabels))');
%     wine_type = net(Features.testFeatures);
%     [~,xxx] = max(wine_type,[],1);
%     plotconfusion(oneHot(Features.testLabels,numClasses,unique(Features.trainingLabels))',oneHot(xxx,numClasses,unique(Features.trainingLabels))');
%     acc.SMMatlab = mean(Features.testLabels(:) == xxx(:));
% outR.predSM_Matlab = xxx;
% outR.testLabels = testLabels;
%     outR.predSM = pred;
% acc.SMMatlab = 0.01;
drawnow
    fprintf('Softmax Accuracy: %0.3f%%\t and AUC: %0.3f%%\n', acc.SM * 100,outR.auc * 100);
    %     perf = mse(testLabels(:), pred(:))
    % mae(testLabels(:)- pred(:))
end

if knn_use
%     Features.trainingLabels = train_label; clear train_label
% Features.trainingFeatures = Train_Features; clear Train_Features
% Features.testFeatures = Test_Features;clear Test_Features
% Features.testLabels = test_label; clear test_label

    Mdl = fitcknn(Features.trainingFeatures',Features.trainingLabels,...
        'NumNeighbors',5,'Standardize',1);
    
    rng(1)
Mdl = fitcknn(Features.trainingFeatures',Features.trainingLabels,...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
    
    [label,score,cost] = predict(Mdl,Features.testFeatures');
end
if rndforest_use
    
end
