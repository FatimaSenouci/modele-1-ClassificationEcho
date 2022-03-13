function I=CustomFcn(filename)

I=imread(filename);

if size(I,3)==1
    I=repmat(I,[1 1 3]);
end

I=imresize(I,[100 100]);
