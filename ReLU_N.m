function out_data = ReLU_N(In_data,n)
% In_data = In_data + randn(size(In_data)).*sqrt(feval(@sigmoidAct, In_data));
out_data = bsxfun(@max,In_data,0);
out_data = bsxfun(@min, out_data,n);

end

