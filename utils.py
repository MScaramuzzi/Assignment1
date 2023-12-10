from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.preprocessing
import random
import os


# setting the seed
def ensure_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def detect_pctg_oovs(oov_arr:np.array, symbol:str) -> float:
    oov_arr_str = oov_arr.astype(str) # Create string from array to check the elements
    oov_arr_len = len(oov_arr)

    # Number case
    if symbol == ".":
        # Check if the symbol is contained in the numpy array and generate a boolean np array of the same dimension
        contains_symbol = np.core.defchararray.find(oov_arr_str, symbol) != -1

        # Perform a filtering given that we are in the case of a number with the comma 
        # and a number (i.e. it contains a digit)
        contains_digit_dot = np.core.defchararray.isdigit(np.core.defchararray.replace(oov_arr_str
                                                        , '.', '')) & contains_symbol
        
        # Compute the division between number of occurances of the dot + number and total length of the oov array
        pctg_oov_digit_dot = len(oov_arr[contains_digit_dot]) / oov_arr_len
        return pctg_oov_digit_dot*100

    # Comma case
    elif symbol == ",":
        contains_symbol = np.core.defchararray.find(oov_arr_str, symbol) != -1
        contains_digit_comma = np.core.defchararray.isdigit(np.core.defchararray.replace(oov_arr_str, ',', '')) & contains_symbol
        pctg_oov_digit_comma= len(oov_arr[contains_digit_comma]) / oov_arr_len
        return pctg_oov_digit_comma*100
    
    # Every other symbol in our case it is only "-" but we left it general
    else:    
        contains_symbol = np.core.defchararray.find(oov_arr_str, symbol) != -1
        pctg_oov_symbol = len(oov_arr[contains_symbol]) / oov_arr_len

    return pctg_oov_symbol*100

def label_encoder(sentences: list[list[str]], tokenizer: keras.preprocessing.text.Tokenizer()) -> list[list[int]]:
  return tokenizer.texts_to_sequences(sentences)

def label_decoder(sentences: list[list[int]], tokenizer: keras.preprocessing.text.Tokenizer()) -> list[list[str]]:
  reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
  words = [[reverse_word_map.get(token) if token!=0 else 'PAD' for token in sentence] for sentence in sentences]
  return words


def generate_weights(voc, vocabulary_dim,embedding_dim,glove_model) -> np.array:
  embedding_weights = np.zeros((vocabulary_dim, embedding_dim))

  embedding_voc = {}

  embedding_weights[1,:] = np.random.uniform(-0.5, 0.5, size=embedding_dim)  # embedding for the OOV token
  for word, index in voc.word2id.items():
    try:
      embedding_weights[index, :] = glove_model[word]
    except (KeyError, TypeError):
      if word in embedding_voc:
        embedding_weights[index, :] = embedding_voc[word]
      if word not in embedding_voc:
        subwords_embedding = {}
        for elem in word.split('-'):
          subwords_embedding.update({elem: np.random.uniform(-0.5, 0.5, size=embedding_dim)})
        for sub_word, _  in subwords_embedding.items():
          try:
            subwords_embedding.update({sub_word: glove_model[sub_word]})  # check if the word is in glove 6B
            subwords_embedding.update({sub_word: embedding_voc[sub_word]})  # check if the word is in the train oov list
          except KeyError:
            pass
        embedding_voc.update({word:(sum(subwords_embedding.values())/len(subwords_embedding))})
      embedding_weights[index, :] = embedding_voc[word]
  return embedding_weights

def get_checkpoint_path(model):
  checkpoint_name = f'{model.name}cp.ckpt'
  checkpoint_path = os.path.join(os.getcwd(), 'Models', checkpoint_name)
  checkpoint_dir = os.path.dirname(checkpoint_path)
  return checkpoint_path

def get_checkpoint(model, monitor: str = "val_loss"):
  filepath = get_checkpoint_path(model)
  cp =  tf.keras.callbacks.ModelCheckpoint(
        filepath = filepath,
        monitor = monitor,
        save_best_only = True,
        save_weights_only = True,
        mode = 'max',
        )
  return cp


def create_folders(seeds: list[int], subfolders: list[str]):
    # our parameters:
    # SEEDS = [42, 1337, 2023]
    # subfolders = ['baseline', 'lstm', 'dense']

    # create the base folder for the checkpoints
    models_path = os.path.join(os.getcwd(), 'Models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    # create subfolders for models and seeds combination
    for seed in seeds:
        for folder in subfolders:
            combined_folder = f"{folder}_{seed}"
            combined_folder_path = os.path.join(models_path, combined_folder)
            if not os.path.exists(combined_folder_path):
                os.makedirs(combined_folder_path)



def get_crisp(pred1D: list[int]) -> list[int]:
    index = np.argmax(pred1D)
    pred_crisp = (np.arange(pred1D.shape[0]) == index).astype('float32')
    return pred_crisp

def get_crisps(pred3D: list[list[list[int]]]) -> list[list[list[int]]]:
    return np.apply_along_axis(get_crisp, -1, pred3D)

def one_hot_reverse(pred:list[list[list[int]]], to_crisp: bool=True) -> np.ndarray:
    if to_crisp:
        pred_crisp = get_crisps(pred)
    pred_class = [[np.argmax(token) for token in sentence] for sentence in pred_crisp]
    return np.array(pred_class)

def decoding_pred(preds:list[list[list[int]]], tokenizer: keras.preprocessing.text.Tokenizer(), to_crisp: bool=True, inverse_one_hot: bool=True) -> np.ndarray:
    if to_crisp:
        pred_crisp = get_crisps(preds)
    if inverse_one_hot:
        pred_class = one_hot_reverse(pred_crisp)
        res = label_decoder(pred_class, tokenizer)
    return np.array(res)


def count_error_per_sentence(y_true:np.ndarray, y_pred:np.ndarray, normalize=False) -> pd.core.frame.DataFrame:
    """
    Misclassified tags are the ones in y_pred that are different from the tags inside y_true
    Returns:
        pd.core.frame.DataFrame: First column sentence number, second column number of errors
    """   

    dict_errors = {}  # store the errors inside a dict
    for idx, sentence in enumerate(y_true):
        mask = sentence != 0 # ignore padding
        true = sentence[mask] 
        pred = y_pred[idx][mask] 
        sum_errors = np.count_nonzero(np.not_equal(true, pred)) # compute number of differences between true and pred then count them
        if normalize: # normalize errors per number of tags in the y_true
            sum_errors=sum_errors/len(true)
        dict_errors[idx] = sum_errors # store the sum in the key correpsonding ot the number of the sentence in the original dataset
        df_error =  pd.DataFrame(list(dict_errors.items()), columns=['num_sentence', 'num_errors'])
        df_error.reset_index(inplace=True)
    return df_error

# Get the predictions masked
def pred_flat(y_true: list[list[int]], y_pred: list[list[int]]) -> tuple[np.ndarray]:
    mask = np.isin(y_true, [0], invert=True)
    y_true_masked = np.array(y_true)[mask]
    y_pred_masked = np.array(y_pred)[mask]
    return y_true_masked, y_pred_masked


### Plotting section ########## 

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# Use FuncFormatter to add percentage sign to y-axis ticks
def percentage_formatter(x, pos):
    return f"{x:.0f}%"

# Barplot
def barplot_label_distr(df_train: pd.DataFrame,df_val: pd.DataFrame) -> None:
    # Generate one row per tag thefore by unpacking the array containing all the labels in a sentence 
    df_expl_tr , df_expl_val = df_train.explode('label') , df_val.explode('label')

    # Count how many times a label occurs in the dataframe 
    label_counts_tr = df_expl_tr['label'].value_counts() 
    label_counts_val = df_expl_val['label'].value_counts()

    # Compute percentage of occurrance of a label
    pctg_lbl_cnts_tr = label_counts_tr / len(df_expl_tr) * 100
    pctg_lbl_cnts_val = label_counts_val / len(df_expl_val) * 100

    # Make color nicer and select colors from palettes
    sns.set(style="whitegrid")
    default_palette = sns.color_palette()


    plt.figure(figsize=(20, 10),dpi=300)
    # Plot the training set bar chart
    plt.bar(label_counts_tr.index, pctg_lbl_cnts_tr, color=default_palette[0], 
            label='Training Set', width=0.4,alpha=0.85)

    # Get the unique labels and their positions
    unique_labels = label_counts_tr.index
    label_positions = np.arange(len(unique_labels))

    # Plot the validation set bar chart next to the training set by offsetting it by 0.4 
    plt.bar(label_positions + 0.4, pctg_lbl_cnts_val.reindex(unique_labels, fill_value=0).values,
            color=default_palette[1], label='Validation Set', width=0.4,alpha=0.85)
    
    plt.legend(fontsize='large')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    plt.gca().xaxis.grid(False)

    plt.xticks(rotation=45, ha='center')
    max_pctg = max(pctg_lbl_cnts_val.max(),pctg_lbl_cnts_tr.max())
    plt.yticks(np.arange(0,max_pctg+2,1),fontsize='medium')
    plt.xlim(-0.5, len(unique_labels) - 0.5)

    plt.ylabel('Percentage of label frequency in dataset (%)',fontsize='large')
    plt.xlabel('Labels',fontsize='large')
    plt.title('Distribution of the labels across train and val',fontsize='x-large')
    plt.show()


def boxplot_sents_len(df_train:pd.DataFrame, df_val:pd.DataFrame, df_test:pd.DataFrame) -> None :
    plt.figure(figsize=(20, 8),dpi=300)

    train_lengths = pd.DataFrame({'sentence_length': df_train['word'].apply(len)})
    val_lengths = pd.DataFrame({'sentence_length': df_val['word'].apply(len)})
    test_lengths = pd.DataFrame({'sentence_length': df_test['word'].apply(len)})

    max_len_train = max(train_lengths['sentence_length'])
    max_len_val = max(val_lengths['sentence_length'])
    max_len_test = max(test_lengths['sentence_length'])


    print(f'Max sentence lenght in the train set: {max_len_train}')
    print(f'Max sentence lenght in the val set: {max_len_val}')
    print(f'Max sentence lenght in the test set: {max_len_test}')

    sns.boxplot(data=[list(train_lengths['sentence_length']),
                    list(val_lengths['sentence_length']),
                    list(test_lengths['sentence_length'])],
            orient= "h",
            );
    xticks_locations = np.arange(0, max(max_len_train, max_len_val,max_len_test) + 10, 20)
    plt.xticks(xticks_locations)
    plt.yticks([0,1,2],['Train','Val','Test'])
    plt.title('Box plot of sentence length across data splits',fontsize="large")
    plt.xlabel("Sentence length [number of words]",fontsize="medium")
    plt.ylabel("Data split",fontsize="medium")
    plt.show()

def plot_history(model_history:dict, model_name:str, metric: str):
    # visualise training history: accuracy
    plt.plot(model_history[model_name].history[metric])
    plt.plot(model_history[model_name].history[f'val_{metric}'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc="lower right")
    plt.show()



from typing import Union
def show_rec_rec_plot(prec, rec, set_type: Union['val','test'], labels_for_metrics: dict):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 7))
    if set_type=='val': 
        plt.suptitle('Validation set', y=1.02, fontsize=16)
    if set_type=='test':
        plt.suptitle('Test set', y=1.02, fontsize=16)

    # Plot precision
    norm_prec = Normalize(vmin=min(prec), vmax=1.3 * max(prec))
    plt.subplot(2, 1, 1)
    if set_type=='val':
        plt.title('Precision scores for non-punctuation labels in validation set')
    if set_type=='test':
        plt.title('Precision scores for non-punctuation labels in test set')
    plt.xlabel('Labels')
    plt.ylabel('Precision')
    plt.xticks(rotation=45, ha='center')
    bars_prec = sns.barplot(x=list(labels_for_metrics.values()), y=prec, palette=sns.color_palette("Spectral", len(prec)))
    for bar, label in zip(bars_prec.patches, prec):
        bar.set_color(plt.cm.Spectral(norm_prec(label)))
    sns.despine(top=True, right=True)

    # Plot recall
    norm_rec = Normalize(vmin=min(rec), vmax=1.3 * max(rec))
    plt.subplot(2, 1, 2)
    if set_type=='val':
        plt.title('Recall scores for non-punctuation labels in validation set')
    if set_type=='test':
        plt.title('Recall scores for non-punctuation labels in test set')
    plt.xlabel('Labels')
    plt.ylabel('Recall')
    plt.xticks(rotation=45, ha='center')
    bars_rec = sns.barplot(x=list(labels_for_metrics.values()), y=rec, palette=sns.color_palette("Spectral", len(rec)))
    for bar, label in zip(bars_rec.patches, rec):
        bar.set_color(plt.cm.Spectral(norm_rec(label)))
    sns.despine(top=True, right=True)

    plt.tight_layout()
    plt.show()