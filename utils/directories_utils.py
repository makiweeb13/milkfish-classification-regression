# Directories
# This file contains the directory paths used in the project.

# Classification directories
train_images = "./dataset/train/images/"
train_labels = "./dataset/train/labels/"
valid_images = "./dataset/valid/images/"
valid_labels = "./dataset/valid/labels/"
test_images = "./dataset/test/images/"
test_labels = "./dataset/test/labels/"

train_output = "./outputs/masks/size_train_masks/"
valid_output = "./outputs/masks/size_valid_masks/"
test_output = "./outputs/masks/size_test_masks/"
data_output = "./outputs/data/"
size_train_data = "train_fish_size_dataframe.csv"
size_valid_data = "valid_fish_size_dataframe.csv"
size_test_data = "test_fish_size_dataframe.csv"

save_gradient_boosting_model = "./models/saved_models/gradient_boosting_classifier.pkl"
save_label_encoder = "./models/saved_models/label_encoder.pkl"

save_random_forest_model = "./models/saved_models/random_forest_classifier.pkl"
save_label_encoder_rf = "./models/saved_models/label_encoder_rf.pkl"

classify_ensemble = "./models/saved_models/classify_ensemble_model.pkl"
classify_svm_meta = "./models/saved_models/svm_meta_model.pkl"

# Regression (Weight) directories
train_weight_images = "./dataset_weight/train/images/"
train_weight_labels = "./dataset_weight/train/labels/"
valid_weight_images = "./dataset_weight/valid/images/"
valid_weight_labels = "./dataset_weight/valid/labels/"
test_weight_images = "./dataset_weight/test/images/"
test_weight_labels = "./dataset_weight/test/labels/"

# train_weight_output = "./outputs/masks/weight_train_masks/"
# valid_weight_output = "./outputs/masks/weight_valid_masks/"
# test_weight_output = "./outputs/masks/weight_test_masks/"
# data_weight_output = "./outputs/data/weight/"
# weight_train_data = "./train_fish_weight_dataframe.csv"
# weight_valid_data = "./valid_fish_weight_dataframe.csv"
# weight_test_data = "./test_fish_weight_dataframe.csv"
