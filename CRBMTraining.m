function [params,Opt,save_file_name] = CRBMTraining(DataSet, Opt,numofLayers,numlayerstart,numdata,wh_method,setnum)
if ~exist('Params','var')
    params = cell(1,numofLayers);
end
%  if numlayerstart == 1
%      error('numlayerstart shoud be grater than 1')
%  end
if numofLayers > 1
    if numlayerstart == 5
        % load Olshousen learned parameters
        X = load('../Params/Params_first_layer_natural_images');
        Opt{1} = X.Opt{1};
        params{1} = X.Params{1};
    elseif numlayerstart >= 2
        L = numofLayers-1;
        load_file_name = sprintf('Params_%dLayer_%s_L%d_%d_%s',L,numdata,L,1,wh_method);
        X = load(sprintf('../Params/%s',load_file_name), 'Opt','params');
        
        for t1 = 1:numlayerstart-1
            Opt{t1} = X.Opt{t1};
            params{t1} = X.params{t1};
        end
    end
end
for LayerNum = numlayerstart : numofLayers
    
    params{LayerNum} = train_crbm_LayerWise(Opt,params,LayerNum,DataSet);
    save_file_name = sprintf('Params_%dLayer_%s_L%d_%d_%s_setnum_%d',LayerNum,numdata,LayerNum,params{LayerNum}.pars.classrbm,wh_method,setnum);
    save(sprintf('../Params/%s',save_file_name), 'Opt','params','-v7.3');
    %         if LayerNum > 1
    %             params = HierarchicalWupdata(DataSet,params,LayerNum-1,c);
    %             save_file_name = sprintf('Params_%dLayer_%s',numofLayers,numdata);
    %             save(sprintf('../Params/%s',save_file_name), 'Opt','params','-v7.3');
    %         end
end
end