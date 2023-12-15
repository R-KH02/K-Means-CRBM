function PAR = GradientSelection (PAR,CRBM,pars)
if strcmpi(pars.RBM_Type,'hyb')
    PAR.dW = PAR.dW_dis  + pars.alpha * PAR.dW_gen;
    PAR.dhbias = PAR.dhbias_dis + pars.alpha * PAR.dhbias_gen + pars.pbias_lambda.* (pars.pbias - CRBM.runningavg_prob);
    PAR.dvbias = PAR.dvbias_dis + pars.alpha * PAR.dvbias_gen;
elseif strcmpi(pars.RBM_Type,'gen')
    PAR.dW = PAR.dW_gen + 1e-3 .* PAR.GradW ;%- 0.0010 .* PAR.GrecondW;  % 1e-6 for 97.4%
    PAR.dhbias = PAR.dhbias_gen + 1e-3 .* PAR.Gradb + pars.pbias_lambda.* (pars.pbias - CRBM.runningavg_prob);
    if strcmp(pars.layerType,'g')
        PAR.dvbias = PAR.dvbias_gen;
    else
        PAR.dvbias = PAR.dvbias_gen ;%- 0.0010 .*PAR.GreconVisBias;
    end
    PAR.dG     = PAR.dGamma;% + pars.pbias_lambda.* (pars.pbias - CRBM.runningavg_prob);
    PAR.dB     =  PAR.dBeta + 0*PAR.Gradb + pars.pbias_lambda.* (pars.pbias - CRBM.runningavg_prob);
    
elseif strcmpi(pars.RBM_Type,'dis')
    PAR.dW = PAR.dW_dis;
    PAR.dhbias = PAR.dhbias_dis + pars.pbias_lambda.* (pars.pbias - mean(CRBM.poshidprobs_dis,[1 2]));
    PAR.dvbias = PAR.dvbias_dis;
end
return
