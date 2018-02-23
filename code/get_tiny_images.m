function features = get_tiny_images(image_paths)

N = size(image_paths, 1);
features = zeros(N, 256);
for a = 1:N    
    curr_im = imread(image_paths{a});
    change_size = imresize(curr_im, [16 16]);
    features(a,:) = reshape(change_size, 1,256);
    features(a,:) = features(a,:) - mean(features(a,:));
    features(a,:) = features(a,:)./norm(features(a,:));
end


