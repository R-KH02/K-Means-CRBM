function [Vdw,Sdw,W] = ADAM_Update(Vdw,Sdw,LR,W,dw,t)
beta1 = 0.9;
beta2 = 0.9;
epsilon = 1e-8;
Vdw = beta1 .* Vdw + (1 - beta1) .* dw;
Sdw = beta2 .* Sdw + (1 - beta2) .* dw.^2;

Vdw_corr = Vdw ./ (1 - beta1 ^ t);
Sdw_corr = Sdw ./ (1 - beta2 ^ t);

W = W + LR .* Vdw_corr ./ sqrt(Sdw_corr + epsilon);
end
