import os
import requests  #  download files
import zipfile, io  #  unzip files
import sys
import pandas as pd
import numpy as np

def download_data(folder,url) -> None: 
    """_summary_

    Args:
        folder (_type_): _description_
        url (_type_): _description_
    """       

    # Create directory if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    req = requests.get(url)
    print("Successful download")

    zip = zipfile.ZipFile(io.BytesIO(req.content))
    zip.extractall(folder)
    zip.close()
    print("Successful extraction")  


def process_files_to_dataframe(file_folder:str) -> pd.DataFrame:
    dataframe_rows = []  # Initialize array containing dataframe rows
    for filename in os.listdir(file_folder): # get the list of all documents in the folder
        file_path = os.path.join(file_folder, filename) # get document path
        try: 
            if os.path.isfile(file_path): 
                with open(file_path, mode='r', encoding='utf-8') as text_file: # Open the file
                    
                    # Gather file id by splitting  ....
                    file_id = filename.split(".")[0].split("_")[1]

                    # Perform data split according to the file number
                    if int(file_id) <= 100:
                        split = 'train'
                    elif int(file_id) <= 150:
                        split = 'val'
                    else:
                        split = 'test'

                    num_sentence = 1  # Counter to keep track of the current sentence
                    words = []
                    labels = []
                    for line in text_file:
                        if line in ['\n', '\r\n']: # Condition for empty line
                            # create single dataframe row with the previous sentence
                            dataframe_row = {
                                "word": words,
                                "label": labels,
                                "file_id": file_id,
                                "num_sentence": num_sentence,
                                "split": split
                            }

                            # Add the current row to the list of sentences of the file
                            num_sentence += 1
                            words = []
                            labels = []
                            dataframe_rows.append(dataframe_row)
                            continue

                        word, label = line.split("\t")[0].lower(), line.split("\t")[1]
                        words.append(word)  # Append to the NumPy array
                        labels.append(label)  # Append to the NumPy array


                    # create single dataframe row
                    dataframe_row = {
                        "word": words,
                        "label": labels,
                        "file_id": file_id,
                        "num_sentence": num_sentence,
                        "split": split
                    }

                    # Add the current row to the list of sentences of the file
                    dataframe_rows.append(dataframe_row)

        except Exception as e:
            print(f'Failed to process {file_path}. Reason: {e}')
            sys.exit(0)

    # Concatenate the list of rows to the DataFrame at once
    df = pd.DataFrame(dataframe_rows)
    # Select only relevant columns
    df = df[["word", "label", "file_id", "num_sentence", "split"]]

    # Sorting the dataframe by file_id and num_sentence
    df = df.sort_values(by=['file_id', 'num_sentence']).reset_index(drop=True)
    
    return df

