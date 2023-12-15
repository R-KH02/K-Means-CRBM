% function im4 = trim_image_for_spacing_fixconv(im3, ws, spacing)
% % % Trim image so that it matches the spacing.
% 
% for i = 1:size(im3,3)
%     im2 = im3(:,:,i);
%                 if mod(size(im2,1)-ws+1, spacing)~=0
%                     n = mod(size(im2,1)-ws+1, spacing);
%                     im2(1:floor(n/2), : ,:) = [];
%                     im2(end-ceil(n/2)+1:end, : ,:) = [];
%                 end
%                 if mod(size(im2,2)-ws+1, spacing)~=0
%                     n = mod(size(im2,2)-ws+1, spacing);
%                     im2(:, 1:floor(n/2), :) = [];
%                     im2(:, end-ceil(n/2)+1:end, :) = [];
%                 end
% % if mod(size(im2,1)-ws+1, spacing)~=0
% %     n1 = spacing - mod(size(im2,1)-ws+1, spacing);
% %     im2 = padarray(im2,[ceil(n1/2),0],'post','replicate');
% %     im2 = padarray(im2,[floor(n1/2),0],'pre','replicate');
% % end
% % if mod(size(im2,2)-ws+1, spacing)~=0
% %     n2 = spacing - mod(size(im2,2)-ws+1, spacing);
% %     im2 = padarray(im2,[0,ceil(n2/2)],'post','replicate');
% %     im2 = padarray(im2,[0,floor(n2/2)],'pre','replicate');
% %     
% % end
%     im4(:,:,i) = im2;
% end
% end
function im2 = trim_image_for_spacing_fixconv(im2, ws, spacing)
% % Trim image so that it matches the spacing.

if mod(size(im2,1)-ws+1, spacing)~=0
    n = mod(size(im2,1)-ws+1, spacing);
    im2(1:floor(n/2), : ,:,:) = [];
    im2(end-ceil(n/2)+1:end, : ,:,:) = [];
end
if mod(size(im2,2)-ws+1, spacing)~=0
    n = mod(size(im2,2)-ws+1, spacing);
    im2(:, 1:floor(n/2), :,:) = [];
    im2(:, end-ceil(n/2)+1:end, :,:) = [];
end
end

