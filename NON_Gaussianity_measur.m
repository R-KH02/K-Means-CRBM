function [Kurtwm,negentropy,entropy] = NON_Gaussianity_measur(W)
% negentw = zeros(1,size(W,3));
% Kurtw = zeros(1,size(W,3));
% moment3 = mean(W(:).^3);
% moment4 = mean(W(:).^4);
% moment2 = mean(W(:).^2);
% kst = moment4 - (3*(moment2.^2));
% negentwm = ((moment3^2)/12) + ((kst^2)/48);

% kk = double(W(:));
% P = hist(kk(:), linspace(0, 1, 256)); 
% P = P(:); P = P(P(:)>0); %we need to delete the value where P = 0, because log(0) = Inf.
% P = P/numel(kk);
% E = -sum(P.*log2(P));

Kurtwm = kurtosis(W(:))-3;
[entropy,negentropy]=mentappr(W(:));
% Kurtwm = kurtosis( W(:));
%
%         [wh,~] = histcounts(W(:),ceil(log2(length(W(:))) + 1));
% negentwm = 0.5*(1 + log(2*pi*var(wh))) - entropy(wh);
% x = 0;
% for i = 1:size(W,3)
%     for j = 1 : size(W,2)
%         x = x+1;
%         w = W(:,j,i);
%         moment3 = mean(w(:).^3);
% moment4 = mean(w(:).^4);
% moment2 = mean(w(:).^2);
% kst = moment4 - (3*(moment2.^2));
% negentw(1,x) = ((moment3^2)/12) - ((kst^2)/48);
% %         [wh,~] = histcounts(w,256);%ceil(log2(length(w(:))) + 1));
%         %         w1 = w - mean2(w);
% %         negentw(x) = 0.5*(1 + log(2*pi*var(wh))) - entropy(wh);
%         Kurtw(1,x) = kurtosis( w(:))-3;
%     end
% end
% Kurtwm = mean(Kurtw);
% negentwm = mean(negentw);
end