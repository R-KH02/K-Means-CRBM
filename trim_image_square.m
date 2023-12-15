function imresp = trim_image_square(imdata,ws,batch_ws,spacing)
% trim the image into batch_ws x batch_ws
[rows, cols, ~] = size(imdata);

rowstart = randi(rows-batch_ws(1)+1);
rowidx = rowstart:rowstart+batch_ws(1)-1;
colstart = randi(cols-batch_ws(2)+1);
colidx = colstart:colstart+batch_ws(2)-1;

imresp = imdata(rowidx, colidx, :,:);
imresp = trim_image_for_spacing(imresp, ws, spacing);

return;


