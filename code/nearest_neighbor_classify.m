function predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)

test_img_count = size(test_image_feats,1);
dimensionality = 20;
nearest_neighbor_dist = pdist2(train_image_feats, test_image_feats,'cosine');
unique_labels = unique(train_labels);
label_size = size(unique_labels, 1);
[~, ind] = sort(nearest_neighbor_dist, 1);
labels = zeros(label_size, test_img_count);
for a = 1:test_img_count
    for b = 1:label_size
        top_labels = train_labels(ind(1:dimensionality, a));
        labels(b,a) = sum(strcmp(unique_labels(b), top_labels));
    end
end
[~, label_ind] = max(labels,[],1);
predicted_categories = unique_labels(label_ind);
end










