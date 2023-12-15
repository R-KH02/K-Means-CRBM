% function [PAR,euclid_dis,S] = dis_gradient_forward (imdata, CRBM,PAR)
% 
% norma = @(a) a./(norm(a(:))+eps);
% ck =  (CRBM.Cluster_centers);
% dck = zeros([size(CRBM.poshidprobs,[1,2]),size(ck),size(CRBM.poshidprobs,4)]);
% dm = zeros([size(CRBM.poshidprobs)]);
% alpha=25;
% for k = 1:size(imdata,4)
%     x = CRBM.poshidprobs(:,:,:,k);
%     for i = 1: size(x,1)
%         for j=1:size(x,2)
%             x1 = x(i,j,:);
%             x2 = permute(x1,[1,3,2]);
%             x1 = norma(x2);  % normalized x1
%             m_ck = bsxfun(@minus, x1,ck);
%             euclid_dis = pdist2(x1, ck);
%             [S,delta_soft] = grad_softmax(euclid_dis,alpha);
%             delta_dis_2 = bsxfun(@times, delta_soft, euclid_dis'.^2);
%             delta_dis = mean(delta_dis_2 ,2);
%             norm_dis = - alpha .* bsxfun(@rdivide,m_ck, euclid_dis');  % - alpha m-ck / || m-ck||2
%             dD_dck = bsxfun(@times, norm_dis, delta_dis );
%             dck(i,j,:,:,k) = dD_dck;
% 
%             % dD_dm ?
%             second_term = 2 .* mean (bsxfun(@times, m_ck, S'),1);
%             dD_dm = - mean(dD_dck,1) + second_term;
%             dD_dm = dD_dm .* (1 - x1.^2)./ (norm(x2(:)+eps));
%             dm(i,j,:,k) = dD_dm;
%         end
%     end
% end
% PAR.dM =  dm;
% PAR.dck = permute(mean(dck,[1,2,5]),[3,4,1,2]);
% 
% delta = CRBM.poshidprobs .* (1 - CRBM.poshidprobs);
% v  = delta .* dm;
% PAR.GradW = convs4(imdata, v, 0);
% PAR.Gradb = permute(mean(v,[4,2,1]),[1,3,2]);
% end


%% ورژن قبلی
function [PAR,euclid_dis,S] = dis_gradient_forward (imdata, CRBM,PAR)


ck =  (CRBM.Cluster_centers);
dck = zeros([size(CRBM.poshidprobs,[1,2]),size(ck),size(CRBM.poshidprobs,4)]);
dm = zeros([size(CRBM.poshidprobs)]);
alpha=25;
for k = 1:size(imdata,4)
    x = CRBM.poshidprobs(:,:,:,k);
    for i = 1: size(x,1)
        for j=1:size(x,2)
            x1 = x(i,j,:);
            x1 = permute(x1,[1,3,2]);
            m_ck = bsxfun(@minus, x1,ck);
            euclid_dis = pdist2(x1, ck);
            [S,delta_soft] = grad_softmax(euclid_dis,alpha);
            delta_dis_2 = bsxfun(@times, delta_soft, euclid_dis'.^2);
            delta_dis = mean(delta_dis_2 ,2);
            norm_dis = - alpha .* bsxfun(@rdivide,m_ck, euclid_dis');  % - alpha m-ck / || m-ck||2
            dD_dck = bsxfun(@times, norm_dis, delta_dis );
            dck(i,j,:,:,k) = dD_dck;

            % dD_dm ?
            second_term = 2 .* mean (bsxfun(@times, m_ck, S'),1);
            dD_dm = - mean(dD_dck,1) + second_term;
            dm(i,j,:,k) = dD_dm;
        end
    end
end
PAR.dM = dm;
PAR.dck = permute(mean(dck,[1,2,5]),[3,4,1,2]);

delta = CRBM.poshidprobs .* (1 - CRBM.poshidprobs);
v  = delta .* dm;
PAR.GradW = convs4(imdata, v, 0);
PAR.Gradb = permute(mean(v,[4,2,1]),[1,3,2]);
end


%%
% norma = @(a) a./norm(a(:));
% % Map to a one channel feature map
% % a two layer Convnet
% x = posprobs;
% % Foraward path
% % x1 = convs(x, CRBM.W00, 0);
% % x2 = ReLU(x1);
% % x3 = convs(x2, CRBM.W01, 0);
% % x4 = ReLU(x3);
% % clustering
% x5 =  x4;%norma(x4);
% x5 = reshape(x5,1,[]);
% euclid_dis = pdist2(CRBM.Cluster_centers, x5);  % vk
% [soft_x] = negsoft(euclid_dis, 25);% alpha = 25 % uk
%
%
% maps = x4;
% m_ci = bsxfun(@minus, x5, CRBM.Cluster_centers);
% dvdm = bsxfun(@times, 2.*( m_ci), soft_x);
%
% dudm = - 25 .* bsxfun(@rdivide, m_ci,euclid_dis);
% dJdai = sum(grad_softmax(soft_x),2);
% dudm = bsxfun(@times,dudm,dJdai .*euclid_dis);% dJdai * dudm * euclid_dis
% dJdm = sum(dvdm + dudm, 1);
% dJdm = reshape(dJdm,size(x4));
%
% dJdck = bsxfun(@times,dudm,-dJdai .*euclid_dis);
% dJdck = 0.1.*sum(-dvdm + dJdck, 2);
% %
%
%
% % Backpropagation
% delta = 0.1.*dJdm;
% delta2 = delta .* (x4 > 0);
% delta3 = convs4(x2, delta2, 0);
% delta4 = delta .* (x2 > 0);
% delta5 = convs4(x, delta4, 0);
%
%
% % update parameters
% t = PAR.t + 1;
% PAR.t = t;
% PAR.dW00 = delta5;%mean(delta5,4);
% PAR.dW01 = delta3;%mean(delta3,4);
% PAR.dck = dJdck;
% [PAR.W00inc,PAR.S00dw,CRBM.W00] = ADAM_Update(PAR.W00inc,PAR.S00dw,1e-4,CRBM.W00,PAR.dW00,t);
% [PAR.W01inc,PAR.S01dw,CRBM.W01] = ADAM_Update(PAR.W01inc,PAR.S01dw,1e-4,CRBM.W01,PAR.dW01,t);
%
% [PAR.ckinc,PAR.Sdck,CRBM.Cluster_centers] = ADAM_Update(PAR.ckinc,PAR.Sdck,1e-4,CRBM.Cluster_centers,PAR.dck,t);
%
%
%
% end