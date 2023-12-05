from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.preprocessing
import random
import os

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

################### 

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

## Set seed

# ensure_reproducbility
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