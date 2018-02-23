function vocab = build_vocabulary( image_paths, vocab_size )

image_count = size(image_paths, 1);
top_count = ceil(10000/image_count);
descriptors = zeros(128, image_count * top_count);
for a=1:image_count
    image = im2single(imread(image_paths{a}));
    points = detectSURFFeatures(image,'MetricThreshold',100);
    features =  extractHOGFeatures(image,points,'NumBins',8,'CellSize',[16 16],'BlockSize',[4 4])';
    descriptors(:,top_count * (a-1) + 1 : top_count * a) = features(:,1:top_count);
end
    
[~,centers] = kmeans(descriptors', vocab_size);
vocab = single(centers);