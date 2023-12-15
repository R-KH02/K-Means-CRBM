function vishidprod21 = tirbm_vishidprod_fixconv(imdata1, H1, ws)

numchannels = size(imdata1,3);
numbases = size(H1,3);
vishidprod21 = zeros(ws,ws,numchannels,numbases,size(imdata1,4));
for i = 1 : size(imdata1,4)
    
    imdata = imdata1(:,:,:,i);
    H = H1(:,:,:,i);
    
    selidx1 = size(H,1):-1:1;
    selidx2 = size(H,2):-1:1;
    vishidprod2 = zeros(ws,ws,numchannels,numbases);
    for c = 1:numchannels
        for b = 1:numbases
            vishidprod2(:,:,c,b) = conv2(imdata(:,:,c), H(selidx1, selidx2, b), 'valid');
        end
    end
    
    vishidprod21(:,:,:,:,i) = vishidprod2;
end
vishidprod21 = reshape(vishidprod21,[ws^2,numchannels,numbases,size(imdata1,4)]);
end