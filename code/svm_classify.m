function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

unique_categories = unique(train_labels); 
categories_count = length(unique_categories);
train_image_count = size(train_image_feats, 1);
test_image_count = size(test_image_feats, 1);
image_size = size(test_image_feats, 2);
Weights = zeros(categories_count, image_size);
Bias = zeros(categories_count, 1);
for a = 1:categories_count
    labels = ones(train_image_count,1).*-1;
    labels(strcmp(unique_categories{a}, train_labels)) = 1;
    SVM = fitcsvm(train_image_feats, labels);
    Weights(a,:) = SVM.Beta';
    Bias(a) = SVM.Bias;
end
confidence = Weights*test_image_feats'+repmat(Bias,1,test_image_count);
[~, ind] = max(confidence);
predicted_categories = unique_categories(ind);
end


