function [CRBM,PAR,pars] = ApplyGrads(CRBM,PAR,pars,avgstart,layerType)


epsilonw = PAR.current_epsilonW;
epsilonb = PAR.current_epsilonB;
t = pars.count;
k = pars.k;

% % PAR.Winc = PAR.Wmomentum*PAR.Winc +   (1-PAR.Wmomentum) * epsilonw  * (PAR.dW);
% % CRBM.W = CRBM.W + PAR.Winc;
% % 
% % PAR.hbiasinc = PAR.Biasmomentum*PAR.hbiasinc +   epsilonb*(1-PAR.Biasmomentum) * (PAR.dhbias);
% % CRBM.hbias_vec = CRBM.hbias_vec +  PAR.hbiasinc;
% % 
% % % if ~strcmpi(layerType,'g')
% %     PAR.vbiasinc = PAR.Biasmomentum*PAR.vbiasinc +   epsilonb*(1-PAR.Biasmomentum) * (PAR.dvbias);
% %     CRBM.vbias_vec = CRBM.vbias_vec +  PAR.vbiasinc;
% % % end

PAR.Winc = PAR.Wmomentum*PAR.Winc + epsilonw *(PAR.dW);
weight = CRBM.W +  (PAR.Winc);
% [PAR.Winc,PAR.Sdw,weight] = ADAM_Update(PAR.Winc,PAR.Sdw,epsilonw,CRBM.W,PAR.dW,t);

% [PAR.vbiasinc,PAR.Sdvis,visBias] = ADAM_Update(PAR.vbiasinc,PAR.Sdvis,epsilonb,CRBM.vbias_vec,PAR.dvbias,t);
PAR.vbiasinc = PAR.Biasmomentum*PAR.vbiasinc + epsilonb* PAR.dvbias;
visBias = CRBM.vbias_vec +  PAR.vbiasinc;

% [PAR.hbiasinc,PAR.Sdhbias,hidBias] = ADAM_Update(PAR.hbiasinc,PAR.Sdhbias,epsilonb,CRBM.hbias_vec,PAR.dhbias,t);
PAR.hbiasinc = PAR.Biasmomentum*PAR.hbiasinc + epsilonb * PAR.dhbias;
hidBias = CRBM.hbias_vec +  PAR.hbiasinc;

% Gamma Update
% PAR.Gammainc = PAR.Biasmomentum*PAR.Gammainc + epsilonb * PAR.dG;
% GammaBias = CRBM.Gamma +  PAR.Gammainc;
[PAR.Gammainc,PAR.SdG,GammaBias] = ADAM_Update(PAR.Gammainc,PAR.SdG,epsilonb,CRBM.Gamma,PAR.dG,t);


% PAR.Betainc = PAR.Biasmomentum*PAR.Betainc + epsilonb * PAR.dB;
% BetaBias = CRBM.Beta +  PAR.Betainc;
[PAR.Betainc,PAR.SdB,BetaBias] = ADAM_Update(PAR.Betainc,PAR.SdB,epsilonb,CRBM.Beta,PAR.dB,t);

% [PAR.ckinc,PAR.Sdck,Cluster_bias] = ADAM_Update(PAR.ckinc,PAR.Sdck,1e-4*epsilonw,CRBM.Cluster_centers,PAR.dck,t);
PAR.ckinc = PAR.Biasmomentum*PAR.ckinc + epsilonb * PAR.dck;
Cluster_bias = CRBM.Cluster_centers +  PAR.ckinc;
if (avgstart > 0 && t > avgstart)
    k = k+1;
    CRBM.W =CRBM.W-(1/k)*(CRBM.W-weight);
    if ~strcmpi(layerType,'g')
        CRBM.vbias_vec = CRBM.vbias_vec-(1/k)*(CRBM.vbias_vec-visBias);
    end
    CRBM.hbias_vec = CRBM.hbias_vec-(1/k)*(CRBM.hbias_vec-hidBias);
    CRBM.Gamma = CRBM.Gamma-(1/k)*(CRBM.Gamma-GammaBias);
    CRBM.Beta = CRBM.Beta - (1/k)* (CRBM.Beta - BetaBias);
    CRBM.Cluster_centers = CRBM.Cluster_centers - (1 / k)*(CRBM.Cluster_centers - Cluster_bias);
else
    CRBM.W = weight;
    if ~strcmpi(layerType,'g')
        CRBM.vbias_vec= visBias;
    end
    CRBM.hbias_vec= hidBias;
    CRBM.Gamma = GammaBias;
    CRBM.Beta = BetaBias;
    CRBM.Cluster_centers = Cluster_bias;
end
return

% % %
% % % function [CRBM,PAR] = ApplyGrads(CRBM,PAR,pars,t,k,avgstart,layerType)
% % %
% % %
% % % epsilon = PAR.current_epsilon;
% % %
% % % PAR.Winc = PAR.momentum*PAR.Winc + (1 - PAR.momentum)*PAR.dW;
% % % weight = CRBM.W + epsilon * (PAR.Winc - pars.l2reg.*CRBM.W);
% % %
% % % PAR.vbiasinc = PAR.momentum*PAR.vbiasinc + (1 - PAR.momentum)*PAR.dvbias;
% % % visBias = CRBM.vbias_vec + epsilon *PAR.vbiasinc;
% % %
% % % PAR.hbiasinc = PAR.momentum*PAR.hbiasinc + (1 - PAR.momentum)*PAR.dhbias;
% % % hidBias = CRBM.hbias_vec + epsilon *PAR.hbiasinc;
% % %
% % % if pars.classrbm
% % %     PAR.Uinc = PAR.momentum*PAR.Uinc + (1 - PAR.momentum)*PAR.dU;
% % %     Uweight = CRBM.U + epsilon *(PAR.Uinc - pars.l2reg.*CRBM.U);
% % %
% % %     PAR.lbiasinc = PAR.momentum*PAR.lbiasinc + (1 - PAR.momentum)*PAR.ddbias;
% % %     labelvec = CRBM.lbias_vec + epsilon *PAR.lbiasinc;
% % % end
% % %
% % % if (avgstart > 0 && t > avgstart)
% % %     k = k+1;
% % %     CRBM.W =CRBM.W-(1/k)*(CRBM.W-weight);
% % %     if ~strcmpi(layerType,'g')
% % %         CRBM.vbias_vec = CRBM.vbias_vec-(1/k)*(CRBM.vbias_vec-visBias);
% % %     end
% % %     CRBM.hbias_vec = CRBM.hbias_vec-(1/k)*(CRBM.hbias_vec-hidBias);
% % %
% % %     if pars.classrbm
% % %             CRBM.U = CRBM.U-(1/k)*(CRBM.U-hidBias);
% % %         CRBM.lbias_vec = CRBM.lbias_vec - (1/k)*(CRBM.lbias_vec - labelvec);
% % %     end
% % % else
% % %     CRBM.W = weight;
% % %     if ~strcmpi(layerType,'g')
% % %         CRBM.vbias_vec= visBias;
% % %     end
% % %     CRBM.hbias_vec= hidBias;
% % %     if pars.classrbm
% % %         CRBM.U = Uweight;
% % %         CRBM.lbias_vec = labelvec;
% % %     end
% % % end
% % % % Qinc = PAR.momentum*Qinc + epsilon*(PAR.dQ - 0.01.*CRBM.Q);
% % % % Qbar = CRBM.Q + Qinc;
% % % % if (avgstart > 0 && t > avgstart)
% % % %     CRBM.Q =CRBM.Q-(1/k)*(CRBM.Q-Qbar);
% % % % else
% % % %     CRBM.Q =  Qbar;
% % % % end
% % % % for k = 1 : size(CRBM.Q,3)
% % % %     CRBM.Q(:,:,k) = CRBM.Q(:,:,k) - diag(diag(CRBM.Q(:,:,k)));
% % % % end
% % % % CRBM.Q(CRBM.Q < 0)=0;
% % %
% % %
% % % return
