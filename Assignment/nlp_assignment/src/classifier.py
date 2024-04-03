from typing import List

import torch

# My imports
from PN_data_processing import data_preprocessing
from PN_dataset import SentimentDataset
from PN_classifier import SentimentClassifier
from PN_trainloop import train, evaluate
from PN_make_prediction import make_predictions

from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import warnings


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
     """



    ############################################# complete the classifier class below
    
    def __init__(self):
        """
        This should create and initilize the model. Does not take any arguments.
        """
        self.model = None
        self.polarity_encoder = LabelEncoder()
        self.tokenizer = None
        self.batch_size = 8

        # silence the futurewarning
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
    
    
    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """

        # Load and preprocess the data
        df_train = data_preprocessing(train_filename)
        #df_dev = data_preprocessing(dev_filename)
        df_train['Polarity'] = self.polarity_encoder.fit_transform(df_train['Polarity'])
        #df_dev['Polarity'] = self.polarity_encoder.transform(df_dev['Polarity'])

        # Setup the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        special_tokens_dict = {'additional_special_tokens': ['[unused0]', '[unused1]']}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # Create the dataset and dataloader
        train_dataset = SentimentDataset(df_train['ModifiedSentence'], df_train['Polarity'], self.tokenizer)
        #dev_dataset = SentimentDataset(df_dev['ModifiedSentence'], df_dev['Polarity'], self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        #dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the model and optimizer
        n_classes = len(self.polarity_encoder.classes_)
        self.model = SentimentClassifier(n_classes=n_classes, tokenizer=self.tokenizer).to(device)
        optimizer = Adam(self.model.parameters(), lr=2e-5)

        # Define the loss function
        class_proportions = {0: 0.26, 1: 0.04, 2: 0.7}
        class_weights = {class_label: (1.0 / proportion) for class_label, proportion in class_proportions.items()}
        weight_sum = sum(class_weights.values())
        num_classes = len(class_weights)
        class_weights = {class_label: (weight / weight_sum) * num_classes for class_label, weight in class_weights.items()}
        weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float32)

        loss_fn = nn.CrossEntropyLoss(weight=weights_tensor).to(device)


        # Train the model
        epochs = 4
        for _ in range(epochs):
            train(self.model, train_loader, loss_fn, optimizer, device)



    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        test_df = data_preprocessing(data_filename)
        test_df['Polarity'] = self.polarity_encoder.transform(test_df['Polarity'])

        test_dataset = SentimentDataset(test_df['ModifiedSentence'], test_df['Polarity'], self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        predictions, _ = make_predictions(self.model, test_loader, device)
        predictions = self.polarity_encoder.inverse_transform(predictions)
        return predictions


