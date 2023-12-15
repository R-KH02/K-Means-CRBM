clear all
clc
% Prepare example data
% Download one image from the van Hateren dataset
% URL: http://bethgelab.org/datasets/vanhateren/
if ~exist('data', 'file'),
    mkdir data
end

if ~exist('data/imk00001.imc', 'file'),
    fprintf('Downloading one image from the van Hateren dataset...');
    urlwrite('http://cin-11.medizin.uni-tuebingen.de:61280/vanhateren/imc/imk00001.imc',...
        'data/imk00001.imc');
        urlwrite('http://cin-11.medizin.uni-tuebingen.de:61280/vanhateren/imc/imk00001.imc',...
        'data/imk00002.imc');
        urlwrite('http://cin-11.medizin.uni-tuebingen.de:61280/vanhateren/imc/imk00001.imc',...
        'data/imk00003.imc');
    fprintf('Done.\n');
end

f1 = fopen('data/imk00001.imc', 'rb', 'ieee-be');
w = 1536; h = 1024;
% resize image and normalize pixels
x = imresize(fread(f1, [w, h], 'uint16'), [w h]/4 + 1);
x = double(x');
x = bsxfun(@rdivide, bsxfun(@minus, x, mean(x,'all')), std(x(:)));
fclose(f1);
data.x(:,:,1,1) = x;
f1 = fopen('data/imk00002.imc', 'rb', 'ieee-be');
w = 1536; h = 1024;

% resize image and normalize pixels
x = imresize(fread(f1, [w, h], 'uint16'), [w h]/4 + 1);
x = double(x');
x = bsxfun(@rdivide, bsxfun(@minus, x, mean(x,'all')), std(x(:)));
data.x(:,:,1,2) = x;
fclose(f1);
f1 = fopen('data/imk00010.imc', 'rb', 'ieee-be');
w = 1536; h = 1024;
% resize image and normalize pixels
x = imresize(fread(f1, [w, h], 'uint16'), [w h]/4 + 1);
x = double(x');
x = bsxfun(@rdivide, bsxfun(@minus, x, mean(x,'all')), std(x(:)));
fclose(f1);
data.x(:,:,1,3) = x;
% Compile mex files
% make(0);

params = getparams;
params.verbose = 4;
[model] = trainCRBM(data, params);