import pandas as pd
import pickle
import time
from os import listdir, makedirs
from os.path import isfile, join, exists


def load_data(data_path="raw_data", cache_file_name="combined_sleep_data", cache_dir_name="cached_data"):
    """
    Loads all files from the given path and puts them into a single file. Performs preprocessing and returns the cleaned
    data as one large dataframe.

    :param cache_dir_name: The name of the directory used to keep cached files.
    :param cache_file_name: The name of the pickle file (without ending) to contain the cached dataframe.
    :param data_path: The path to the directory containing the raw input data.
    :return: A dataframe containing the concatenated sleep records.
    """
    cache_file = f"{cache_dir_name}/{cache_file_name}.pkl"

    if not exists(cache_dir_name):
        makedirs(cache_dir_name)

        # find files in input directory
        data_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        print(f"Found {len(data_files)} input files in directory '{data_path}'.")

        # load and concatenate input data
        dataframes = []
        for file in data_files:
            dataframes.append(pd.read_csv(f"{data_path}/{file}"))
        data_concat = pd.concat(dataframes, ignore_index=True)

        # run preprocessing and return data
        data = preprocess_data(data_concat)
        print(f"Combined input data has {data.shape[0]} entries.")

        # dump data to a pickle file for caching
        with open(cache_file, "wb") as file:
            pickle.dump(data, file)
    else:
        # load existing pickle file
        with open(cache_file, "rb") as file:
            data = pickle.load(file)

    return data


def preprocess_data(data):
    """
    Performs preprocessing steps on the data. The steps are:
        1) Extract time and day from the timestamp and put them into new columns. Remove timestamp column.
        2) Remove entries with NaN values.
        3) Perform outlier detection.

    :param data: The dataframe containing combined input data.
    :return: A dataframe with the preprocessed data.
    """
    data["timestamp"] = data["timestamp"].astype("datetime64[ns]")
    data["day"] = data.timestamp.dt.day
    data["time"] = data.timestamp.dt.time
    data.drop("timestamp", axis=1, inplace=True)

    return data


if __name__ == '__main__':
    start = time.time()
    d = load_data()
    print(f"{time.time()-start:.2f} seconds elapsed for data loading/preprocessing.")
