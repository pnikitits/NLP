import pandas as pd



def data_preprocessing(data_filename):
    column_names = ['Polarity', 'Aspect', 'TargetTerm', 'Offsets', 'Sentence']
    df = pd.read_csv(data_filename, sep='\t', header=None, names=column_names)
    df['ModifiedSentence'] = df.apply(lambda row: highlight_target_term(row['Sentence'], row['TargetTerm'], row['Offsets']), axis=1)

    df = df.drop(columns=['TargetTerm', 'Offsets', 'Sentence' , 'Aspect'])
    
    return df



def highlight_target_term(sentence, target_term, offsets, crop_chars=40):
    """
    Inserts special tokens around the target term in the sentence based on offsets and crops the sentence.
    
    Args:
    - sentence (str): The sentence from the data.
    - target_term (str): The target term to be highlighted.
    - offsets (str): The start and end offsets of the target term in the sentence.
    - crop_chars (int): The number of characters to include before and after the target term.
    
    Returns:
    - modified_sentence (str): The sentence with the target term highlighted and cropped.
    """
    start, end = map(int, offsets.split(':'))
    # Crop the sentence to a fixed window size around the target term
    cropped_sentence = sentence[max(0, start-crop_chars):min(len(sentence), end+crop_chars)]
    # Insert special tokens around the target term
    highlighted_sentence = (cropped_sentence[:start-max(0, start-crop_chars)] + 
                            " [unused0] " + cropped_sentence[start-max(0, start-crop_chars):end-max(0, start-crop_chars)] + 
                            " [unused1] " + cropped_sentence[end-max(0, start-crop_chars):])
    return highlighted_sentence