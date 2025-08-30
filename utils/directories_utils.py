# Directories
# This file contains the directory paths used in the project.

# Classification directories
train_images = "./dataset/train/images/"
train_labels = "./dataset/train/labels/"
valid_images = "./dataset/valid/images/"
valid_labels = "./dataset/valid/labels/"
test_images = "./dataset/test/images/"
test_labels = "./dataset/test/labels/"

size_train_cropped = "./outputs/cropped/size_train_cropped/"
size_valid_cropped = "./outputs/cropped/size_valid_cropped/"
size_test_cropped = "./outputs/cropped/size_test_cropped/"

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
train_weight_images = "./weight_dataset/train/images/"
train_weight_labels = "./weight_dataset/train/labelTxt/"
valid_weight_images = "./weight_dataset/valid/images/"
valid_weight_labels = "./weight_dataset/valid/labelTxt/"
test_weight_images = "./weight_dataset/test/images/"
test_weight_labels = "./weight_dataset/test/labelTxt/"

train_weight_output = "./outputs/masks/weight_train_masks/"
valid_weight_output = "./outputs/masks/weight_valid_masks/"
test_weight_output = "./outputs/masks/weight_test_masks/"
weight_train_data = "train_fish_weight_dataframe.csv"
weight_valid_data = "valid_fish_weight_dataframe.csv"
weight_test_data = "test_fish_weight_dataframe.csv"
regressor_gradient_boosting_model = "./models/saved_models/gradient_boosting_regressor.pkl"
regressor_random_forest_model = "./models/saved_models/random_forest_regressor.pkl"

regress_ensemble = "./models/saved_models/regress_ensemble_model.pkl"

saved_class_scaler = "./models/saved_models/class_scaler.pkl"
saved_regress_scaler = "./models/saved_models/regress_scaler.pkl"