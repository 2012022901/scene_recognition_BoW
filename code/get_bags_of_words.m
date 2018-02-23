
function image_feats = get_bags_of_words(image_paths)

load('vocab.mat')
vocab_size = size(vocab, 1);
image_count = size(image_paths, 1);
image_feats = zeros(image_count, vocab_size);
for a = 1:image_count
    img = im2single(imread(image_paths{a}));
    points = detectSURFFeatures(img,'MetricThreshold',100);
    features =  extractHOGFeatures(img,points,'NumBins',8,'CellSize',[16 16],'BlockSize',[4 4])';
    features = single(features)';
    [ind, ~] = knnsearch(vocab, features);
    image_feats(a,:) = histc(ind, 1:vocab_size)';
end
end




