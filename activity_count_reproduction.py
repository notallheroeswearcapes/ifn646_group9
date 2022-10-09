import warnings
from preprocess import load_data, inform
warnings.filterwarnings('ignore')

train, test, val = load_data()

inform(train)

inform(test)

inform(val)

