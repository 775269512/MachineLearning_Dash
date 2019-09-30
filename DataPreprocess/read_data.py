input_path = "../input/"
os.listdir(input_path)

train = pd.read_csv(input_path + "train.csv", nrows=None, usecols=None)
test = pd.read_csv(input_path + "test.csv", nrows=None, usecols=None)
sub = pd.read_csv(input_path + "submission.csv")

print(train.shape, test.shape, sub.shape)

id = ""
target = ""