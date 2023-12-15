% clc
warning off images:imshow:magnificationMustBeFitForDockedFigure
close all
clear

addpath(genpath('whitening'));
addpath(genpath('maxpooling'));
addpath(genpath('minFunc'));
if ~exist(sprintf('../Params'),'dir')
    mkdir('../Params')
end
if ~exist(sprintf('../Data_L'),'dir')
    mkdir('../Data_L')
end
wh_method = 'ZCA_GCN';
numofLayers = 1;
numlayerstart = 1;
MinibatchSize = 16;


dataname = 'MNIST';
setnum = 1;
% load('mnist_small.mat');
if 0
    [imgDataTrain, labelsTrain, imgDataTest, labelsTest] = prepareMNISTData;
    imgDataTrain = im2double(imgDataTrain);
    labelsTrain = grp2idx(labelsTrain);
    for k = 1:size(imgDataTrain,4)
        for k1 = 1:size(imgDataTrain,3)
            im2 = imgDataTrain(:,:,k1,k);
            m = mean(mean(im2));
            im2 = im2 - m;
            s = std(im2(:));
            im2 = im2./(1*s);
            imgDataTrain(:,:,k1,k) = sqrt(0.1) .* im2;
        end
    end
    
    imgDataTest = im2double(imgDataTest);
    labelsTest = grp2idx(labelsTest);
    
    for k = 1:size(imgDataTest,4)
        for k1 = 1:size(imgDataTest,3)
            im2 = imgDataTest(:,:,k1,k);
            m = mean(mean(im2));
            im2 = im2 - m;
            s = std(im2(:));
            im2 = im2./(1*s);
            imgDataTest(:,:,k1,k) = sqrt(0.1) .* im2;
        end
    end
    RGB_channel = 1;
    
    train_lbls = labelsTrain;%   classes(indxTrain(1,:));
    nclasses = length(unique(train_lbls));
    pp = 5/6;
    Xtr_training = [];   Xtr_validation = [];
    l_tr_training = [];  l_tr_validation = [];
    for cc=1:nclasses
        curr_X = imgDataTrain(:,:,:,find(train_lbls == cc));
        currNc = size(curr_X,4);
        qq = randperm(currNc);
        newNc = floor(pp * currNc);
        Xtr_training = cat(4,Xtr_training, curr_X(:,:,:,qq(1:newNc)));
        Xtr_validation = cat(4,Xtr_validation, curr_X(:,:,:,qq(newNc+1:end)));
        l_tr_training = cat(2,l_tr_training, cc*ones(1,newNc));
        l_tr_validation = cat(2,l_tr_validation, cc*ones(1,currNc-newNc));
    end
    indxtrain = randperm(size(Xtr_training,4));
    indxvalidation = randperm(size(Xtr_validation,4));
    
    DataSet.XTrain =  Xtr_training(:,:,:,indxtrain);%trainData(:,:,:,1:end-1000);%
    DataSet.YTrain =double(l_tr_training(indxtrain));% double(labels(1:end-1000));%
    DataSet.ClsInfo = DataSet.YTrain ;
    DataSet.XValidation = Xtr_validation(:,:,:,indxvalidation);%trainData(:,:,:,end-999:end);%
    DataSet.YValidation = double(l_tr_validation(indxvalidation));%double(labels(end-999:end));%
    DataSet.XTest = imgDataTest;%permute(testData,[1,2,4,3]);
    DataSet.YTest = labelsTest;
    save MNIST_large DataSet
else
    
    load('MNIST_large.mat')
end
DataSet.YTrain_onehot = oneHot(DataSet.YTrain);
DataSet.YValidation_onehot = oneHot(DataSet.YValidation);
DataSet.YTest_onehot = oneHot(DataSet.YTest);


%%
if 1
dataname = 'MNIST';
setnum = 1;
Opt = makeCRBMParameters(numofLayers,MinibatchSize,dataname);

    [params,Opt,file_name] = CRBMTraining(DataSet, Opt,numofLayers,numlayerstart,dataname,wh_method,setnum);
    acc = Feature_Extraction(file_name,DataSet,dataname,'test');
end






if 0
    params = cell(1,numofLayers);
    A = cell(1,numofLayers);
    c = 0;
    for fs1 = [:]
        Opt{1, 1}.ws = fs1;
        for st1 = [:]
            Opt{1, 1}.pbias = st1;
            for pbias_lb = [:]
                Opt{1, 1}.pbias_lb = pbias_lb;
                for beta = [:]
                    Opt{1, 1}.beta = beta;
                    for L2reg1 = [:]
                        Opt{1, 1}.l2reg = L2reg1;
                        Opt{1,1}.classrbm = 1;
                        c = c + 1;
                        LayerNum = 1;
                        params{LayerNum} = train_crbm_LayerWise(Opt,params,LayerNum,DataSet);
                        file_name = sprintf('Params_%dLayer_%d_setnum_%d_label_%d',LayerNum,c,setnum,Opt{1,1}.classrbm);
                        save(sprintf('../Params/%s',file_name), 'Opt','params','-v7.3');
                        A{setnum,c} = Feature_Extraction(file_name,DataSet,dataname,'valid_mode');
                        A{setnum,c}.filterSize = fs1;
                        A{setnum,c}.sparsityTarget = st1;
                        %                         A{setnum,c}.pbias_lb = pbias_lb;
                        %                         A{setnum,c}.beta = beta;
                        A{setnum,c}.l2reg = L2reg1;
                        save(sprintf('../Params/%s',file_name), 'Opt','params','A','-v7.3');
                    end
                end
            end
        end
    end
end
