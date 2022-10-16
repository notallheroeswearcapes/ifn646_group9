import pandas as pd
import numpy as np
import pickle
import time
import random
from os import listdir, makedirs
from os.path import isfile, join, exists
import warnings

warnings.filterwarnings('ignore')

DATA_SETS = ["Full data", "Training data", "Test data"]


def load_data(
        data_path="raw_data",
        cache_file_name="combined_sleep_data",
        cache_dir_name="cached_data"
):
    """
    Loads all files from the given path. Performs preprocessing and returns the cleaned data as three DataFrames:
        training data, test data, and validation data. Extracts time and day from the timestamp column and puts them
        into new columns. Removes timestamp column. Splits data into training, test, and validation sets.

    :param cache_dir_name: The name of the directory used to keep cached files.
    :param cache_file_name: The name of the pickle file (without ending) to contain the cached DataFrame.
    :param data_path: The path to the directory containing the raw input data.
    :return: Three DataFrames containing the concatenated sleep records separated into full, train, test, and validation
        sets.
    """
    cache_file_names = [
        f"{cache_dir_name}/{cache_file_name}_full.pkl",
        f"{cache_dir_name}/{cache_file_name}_train.pkl",
        f"{cache_dir_name}/{cache_file_name}_test.pkl"
    ]

    if not exists(cache_dir_name):
        makedirs(cache_dir_name)

    if __cache_files_exist(cache_file_names):
        # load existing pickle files
        print("Loading cached files.")
        data = []
        for file in cache_file_names:
            with open(file, "rb") as f:
                data.append(pickle.load(f))
    else:
        # find files in input directory
        data_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        print(f"Found {len(data_files)} input files in directory '{data_path}'.")

        # load and concatenate input data
        dataframes = []
        for day, file in enumerate(data_files):
            frame = pd.read_csv(f"{data_path}/{file}")
            frame.insert(0, 'day', day + 1)
            dataframes.append(frame)
        data_concat = pd.concat(dataframes, ignore_index=True)

        # day/time extraction
        data_concat["timestamp"] = pd.to_datetime(data_concat["timestamp"], format="%d/%m/%Y %H:%M:%S")
        data_concat["time"] = data_concat.timestamp.dt.time
        data_concat.drop("timestamp", axis=1, inplace=True)

        print(f"Data has {data_concat.shape[0]} entries before preprocessing.\n")

        # split data into training, test, validation sets
        training_data, test_data = __split_data(data_concat)

        # run preprocessing and pickle data
        data = [data_concat, training_data, test_data]
        for data_set, name, file in zip(data, DATA_SETS, cache_file_names):
            with open(file, "wb") as f:
                processed = __preprocess_data(data_set, name)
                print(f"{name} has {processed.shape[0]} entries after preprocessing.\n")
                pickle.dump(processed, f)

    return data


def inform(df):
    """
    Prints some information about the given DataFrame.

    :param df: The DataFrame to analyse.
    """
    print(f"Shape of data: {df.shape}")
    cl_0 = (df["Actiware classification"] == 0).sum()
    ac_0 = (df["Actiwatch activity counts"] == 0).sum()
    print(f"There are {(cl_0 / len(df)) * 100.0:.2f}% 0 values in column 'Actiware classification'.")
    print(f"There are {(ac_0 / len(df)) * 100.0:.2f} 0 values in column 'Actiwatch activity counts'.")


def __split_data(df, num_train=22, num_test=5):
    """
    Splits the combined DataFrame into training and test sets. Takes a random set of days into account per set that can
    be specified via parameters.

    :param df: The original combined DataFrame.
    :param num_train: The number of days to include in the training set.
    :param num_test: The number of days to include in the test set.
    :return: Two DataFrames for training and test sets.
    """
    if num_train + num_test > 27:
        raise Exception("Ratio for train/test split is incorrect!")
    seq = range(1, 28)
    random.seed(123456)

    train_days = random.sample(seq, k=num_train)
    seq = [x for x in seq if x not in train_days]
    test_days = random.sample(seq, k=num_test)

    train = df[df["day"].isin(train_days)]
    test = df[df["day"].isin(test_days)]

    return train, test


def __preprocess_data(data, name):
    """
    Performs preprocessing steps on the data. The steps are:
        1) Perform outlier detection.
        2) Handle entries with NaN values.

    :param name: The name of the DataFrame.
    :param data: The DataFrame containing input data.
    :return: The DataFrame with the preprocessed data.
    """

    print(f"===== {name} =====")

    # outlier detection, replace outliers with median
    outlier_detection_set = data
    potential_outliers = __detect_outliers(outlier_detection_set)
    potential_outlier_cols = potential_outliers.sum()[potential_outliers.sum() > 0]
    potential_outlier_cols = potential_outlier_cols * 100 / len(outlier_detection_set)
    pot_out_perc = potential_outliers.sum() * 100 / len(outlier_detection_set)
    print(f"{len(potential_outlier_cols)} of {outlier_detection_set.shape[1]} columns contain possible outliers.")
    print(f"{pot_out_perc.mean():.2f}% of the data are considered a potential outlier and are replaced by the mean.")
    print(potential_outlier_cols.sort_values(ascending=False))
    removed_outliers = outlier_detection_set.mask(potential_outliers, outlier_detection_set.median(), axis=1)
    # data = removed_outliers

    # missing data analysis
    print('\nMissing values for columns:')
    missing_values, mean_mv = __missing_data_analysis(data)
    print(f"Percentage of missing values in merged sleep data: {mean_mv:.2f}%")
    print("\nMissing data per columns in percent:")
    print(missing_values)

    # handle missing values according to supplied strategy
    data = __handle_missing_values(data, 'Actiware classification')

    return data


def __handle_missing_values(df, classification_col):
    """
    Missing values are handled according to the note for actigraphy in the assignment specification. The first and last
    5-minute intervals of uninterrupted sleep per day are extracted and all classification values before or after that
    are set to 1 (i.e. awake). Rows where both the columns 'Actiwatch activity counts' and 'Actiware classification' are
    safely removed.

    :param df: The DataFrame containing the (sub)set of data.
    :param classification_col: The column containing classification values.
    :return: The DataFrame with handled missing values.
    """
    # remove rows where both activity counts and classification are missing
    length_before = len(df)
    df = df[~(df['Actiwatch activity counts'].isna() & df['Actiware classification'].isna())]
    days = df['day'].unique()
    indices = []
    for day in days:
        subset = df[df['day'] == day]
        indices.append(__find_first_and_last_uninterrupted_sleep_intervals(subset, classification_col))

    rows_altered = 0
    for start, first_sleep, last_sleep, end in indices:
        rows_altered += first_sleep - start + end - last_sleep
        df.loc[start:first_sleep + 1, classification_col] = 1
        df.loc[last_sleep:end + 1, classification_col] = 1

    print(f"""
    {length_before - len(df)} rows were dropped where both activity counts and classification were missing.
    That is roughly {((length_before - len(df)) / length_before) * 100.0:.2f}% of the dataset.""")
    print(f"""
    {rows_altered} classifications were set to 1 for the first and last 5 minutes of uninterrupted sleep.
    That is roughly {(rows_altered / len(df)) * 100.0:.2f}% of the dataset.""")
    return df


def __find_first_and_last_uninterrupted_sleep_intervals(df, classification_col):
    """
    Extracts the first and last 5-minute intervals of uninterrupted sleep (i.e. continuous 0 values) from the
    classification column.

    :param df: The sub-DataFrame containing actigraphy for a single day to check for sleep intervals.
    :param classification_col: The column containing classification values.
    :return: A 4-tuple: the start index of the sub-DataFrame, the index of the beginning of the first 5-minute interval,
    the index of the end of the last 5-minute interval and the end index of sub-DataFrame.
    """
    indices = df.index[df[classification_col] == 0].tolist()
    first_sleep = np.nan
    last_sleep = np.nan
    for i, index in enumerate(indices):
        interval = list(range(index, index + 21))
        if indices[i:i + 21] == interval:
            first_sleep = index
            break
    for i, index in enumerate(reversed(indices)):
        interval = list(range(index - 20, index + 1))
        comparison_interval = indices[-i - 21:None if i == 0 else -i]
        if comparison_interval == interval:
            last_sleep = index
            break
    if first_sleep == np.nan or last_sleep == np.nan:
        raise Exception("Something went wrong while extracting the first and last sleep indices.")
    return df.index[0], first_sleep, indices[-1], df.index[-1]


def __detect_outliers(df, iqr_factor=4):
    """
    Performs an outlier detection with regard to an interval of a multiple of the IQR.

    :param df: The DataFrame to check for outliers.
    :param iqr_factor: The factor for the IQR.
    :return: A boolean mask indicating outliers where True.
    """
    q1s = df.quantile(0.25)
    q3s = df.quantile(0.75)
    iqr = (q3s - q1s) + 1
    lo = q1s - iqr_factor * iqr
    hi = q3s + iqr_factor * iqr
    outliers_mask = (df <= lo) | (df >= hi)
    return outliers_mask


def __missing_data_analysis(df):
    """
    Checks a given DataFrame for missing values. Returns a DataFrame with the percentage of missing values per column.

    :param df: The original DataFrame to check.
    :return: A DataFrame containing the percentage of missing values per column and the mean percentage of missing
    values of the whole data.
    """
    perc_missing = df.isnull().sum() * 100 / len(df)
    mv = pd.DataFrame({'column_name': df.columns, 'percent_missing': perc_missing})
    mv.sort_values(by="percent_missing", ascending=False, inplace=True)
    mv.reset_index(drop=True, inplace=True)
    return mv, mv['percent_missing'].mean()


def __cache_files_exist(cache_file_names):
    for file in cache_file_names:
        if not exists(file):
            return False
    return True


if __name__ == '__main__':
    start_time = time.time()
    full_d, training_d, test_d = load_data()
    print(f"{time.time() - start_time:.2f} seconds elapsed for data loading/preprocessing.")
