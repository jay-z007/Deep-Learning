from data_utils import *
from DCNN_wrapper import *

train_dataset, train_labels, test_dataset, test_labels = load_data("./data")

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
				shuffle_split_data(train_dataset, train_labels, test_dataset, test_labels, valid_size=0.2)

model = DCNN_model(valid_dataset, test_dataset, test_labels)

#sess = model.restore_last_session()

model.train(train_dataset, train_labels, valid_dataset, valid_labels)

