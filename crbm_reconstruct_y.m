function labelProbs = crbm_reconstruct_y(S, CRBM)

temp1 = squeeze(sum(sum(S,1),2));
% % % % % % % % % % % % % temp2 = exp (temp1' * RBM.U +  RBM.dbias');
% % % % % % % % % % % % % temp3 = sum(temp2,2);
% % % % % % % % % % % % % 
% % % % % % % % % % % % % labelProbs = temp2 ./ temp3;
% % % % % % % % % % % % x = temp1' * U +  lbias_vec';
% % % % % % % % % % % % 
% % % % % % % % % % % % labelProbs = 1 ./ (1 + exp(-x));
% % % % % % % % % % % % labelProbs = labelProbs ./ (sum(labelProbs));

x = (temp1' * CRBM.U +  CRBM.lbias_vec')';%(1/pars.std_gaussian^2).* 
x = bsxfun(@minus,x,max(x,[],1));
labelProbs = exp(x);
labelProbs = bsxfun(@rdivide, labelProbs, sum(labelProbs, 1))';

% labelStates = softmax_sample ( labelProbs_all); %sample label vector

end


