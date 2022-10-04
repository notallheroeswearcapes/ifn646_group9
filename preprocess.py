import pandas as pd
import pickle
import time
import random
from os import listdir, makedirs
from os.path import isfile, join, exists


def load_data(data_path="raw_data", cache_file_name="combined_sleep_data", cache_dir_name="cached_data"):
    """
    Loads all files from the given path. Performs preprocessing and returns the cleaned data as three DataFrames:
        training data, test data, and validation data.

    :param cache_dir_name: The name of the directory used to keep cached files.
    :param cache_file_name: The name of the pickle file (without ending) to contain the cached DataFrame.
    :param data_path: The path to the directory containing the raw input data.
    :return: Three DataFrames containing the concatenated sleep records separated into train, test, and validation sets.
    """
    cache_file_train = f"{cache_dir_name}/{cache_file_name}_train.pkl"
    cache_file_test = f"{cache_dir_name}/{cache_file_name}_test.pkl"
    cache_file_val = f"{cache_dir_name}/{cache_file_name}_val.pkl"

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
        training_data, test_data, validation_data = __preprocess_data(data_concat)
        print(f"Training data has {training_data.shape[0]} entries after preprocessing.")
        print(f"Test data has {test_data.shape[0]} entries after preprocessing.")
        print(f"Validation data has {validation_data.shape[0]} entries after preprocessing.")

        # dump data to a pickle file for caching
        with open(cache_file_train, "wb") as file:
            pickle.dump(training_data, file)
        with open(cache_file_test, "wb") as file:
            pickle.dump(test_data, file)
        with open(cache_file_val, "wb") as file:
            pickle.dump(validation_data, file)
    else:
        # load existing pickle file
        print("Loading cached files.")
        with open(cache_file_train, "rb") as file:
            training_data = pickle.load(file)
        with open(cache_file_test, "rb") as file:
            test_data = pickle.load(file)
        with open(cache_file_val, "rb") as file:
            validation_data = pickle.load(file)

    return training_data, test_data, validation_data


def inform(df):
    """
    Prints some information about the given DataFrame.

    :param df: The DataFrame to analyse.
    """
    print(f"Shape of data: {df.shape}")
    cl_0 = (df["Actiware classification"] == 0).sum()
    ac_0 = (df["Actiwatch activity counts"] == 0).sum()
    print(f"There are {(cl_0/len(df))*100.0:.2f}% 0 values in column 'Actiware classification'.")
    print(f"There are {(ac_0/len(df))*100.0:.2f} 0 values in column 'Actiwatch activity counts'.")


def __preprocess_data(data):
    """
    Performs preprocessing steps on the data. The steps are:
        1) Extract time and day from the timestamp and put them into new columns. Remove timestamp column.
        2) Perform outlier detection.
        3) Remove entries with NaN values.
        4) Split into training, test, validation sets. Remove day column.

    :param data: The DataFrame containing combined input data.
    :return: Three DataFrames with the preprocessed data split into the relevant sets.
    """

    # day/time extraction
    data["timestamp"] = data["timestamp"].astype("datetime64[ns]")
    data["day"] = data.timestamp.dt.day
    data["time"] = data.timestamp.dt.time
    data.drop("timestamp", axis=1, inplace=True)

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
    data = removed_outliers

    # missing data analysis
    print('\nMissing values for columns:')
    missing_values, mean_mv = __missing_data_analysis(data)
    print(f"Percentage of missing values in merged sleep data: {mean_mv:.2f}%")
    print("\nMissing data per columns in percent:")
    print(missing_values)

    len_before = len(data)
    data = data.dropna().reset_index(drop=True)
    print(f"{100-(len(data)/len_before)*100.0:.2f}% of the {len_before} entries were dropped.")

    # split data into training, test, validation sets
    training, test, validation = __split_data(data)

    return training, test, validation


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


def __split_data(df, num_train=20, num_test=5):
    """
    Splits the combined DataFrame into training, test, validation sets. Takes a random set of days into account per
    set that can be specified via parameters.

    :param df: The original combined DataFrame.
    :param num_train: The number of days to include in the training set.
    :param num_test: The number of days to include in the test set.
    :return: Three DataFrames for training, test, validation sets.
    """
    if num_train + num_test >= 27:
        raise Exception("Ratio for train/test/validation split is incorrect!")
    seq = range(1, 28)
    random.seed(123456)

    train_days = random.sample(seq, k=num_train)
    seq = [x for x in seq if x not in train_days]
    test_days = random.sample(seq, k=num_test)
    val_days = [x for x in seq if x not in test_days]

    train = df[df["day"].isin(train_days)]
    test = df[df["day"].isin(test_days)]
    val = df[df["day"].isin(val_days)]

    train.drop("day", axis=1, inplace=True)
    test.drop("day", axis=1, inplace=True)
    val.drop("day", axis=1, inplace=True)

    return train, test, val


if __name__ == '__main__':
    start = time.time()
    d = load_data()
    print(f"{time.time()-start:.2f} seconds elapsed for data loading/preprocessing.")
