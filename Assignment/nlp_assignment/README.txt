1. Pierre Nikitits

2. Description of the implemented classifier:

    Data preprocessing:
        - Highlight Target Term: I apply the highlight_target_term function to each row,
        which enhances the sentence by highlighting the target term. This is done by inserting special
        tokens [unused0] and [unused1] before and after the term.
        - Cropping the sentence a defined number of characters on each side of the target term, this removes
        extra information which might not be relevant to the target
        - Encoding of the target column: 0,1,2 for negative,neutral,positive respectively.
        - I pass only the encoded target and the precessed sentence to the classifier.

    Classifier:
        - The base of the classifier is a BERT (bert-base-uncased) which I resize the token embedding
        to work with the extra token (used to highlight the target terms).
        - I added 2 dropout layers to prevent overfitting.
        - I used a fully connected linear layer to reduce the dimensionality of the output of
        the BERT model by half.
        - I then use a ReLU activation function to add non-linearity.
        - The layer normalisation is done to stabalise the learning and improve convergence.
        - The final layer maps the reduced hidden size to the number of classes

        Forward pass: the computation that is done when the model is called.


3. Accuracy on the dev dataset:
    Mean Dev Acc.: 83.03 (0.62)
    Dev accs: [83.51, 83.78, 82.45, 82.18, 83.24]