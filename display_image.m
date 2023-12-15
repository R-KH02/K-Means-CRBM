
function I = display_image(in_d)
%% 
s = size(in_d,3);

s1 = ceil(sqrt(s));

for i = 1:s
    subplot(s1,s1,i),
    imshow(in_d(:,:,i),[]);
end
end