from lib import data, model, common
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

PATH = "/home/chris/projects/Kaggle/heart_200719/data/heart.csv"
NORMAL_COL_NAME = ["age", "trestbps", "chol", "thalach", "oldpeak"]

PRINT_STEP = 3000
VALIDATION_STEP = 30000
BATCH_SIZE = 32
step = 0
lr = 0.0001

# read csv and created a clean df, then normalized specific columns
df = data.read_csv(PATH)
df_normalised = None
for i, key in enumerate(df.keys()):
    if key in NORMAL_COL_NAME:
        df_normalised_col = data.normalize_data(df.loc[:,key], key)
    else:
        df_normalised_col = df.loc[:,key]
    if i == 0:
        df_normalised = df_normalised_col
    else:
        df_normalised = data.append_col(df_normalised_col, df_normalised)

# change from dataframe to array, shape = (304, 14)
arr = data.df2array(df_normalised)

# shuffle rows array
arr_shuffled = data.shuffle(arr)

# get the training and testing set
train_set, test_set = data.split_data(arr_shuffled, percentage=0.8)

# define the net
net = model.simple_net(input_size=13)

# define optimizer
optimizer = optim.Adam(net.parameters(), lr=lr)

# define writer
now = datetime.now()
dt_string = now.strftime("%y%m%d_%H%M%S")
RUNS_SAVE_PATH = "../docs/2/runs/" + dt_string
writer = SummaryWriter(log_dir=RUNS_SAVE_PATH, comment="heart_attack")

# tester
validator = common.Validator(net, writer)

losses = []
test_losses = []
while True:

    net.train()
    optimizer.zero_grad()

    # train_x, train_y
    x, y = data.batch_gen(train_set, BATCH_SIZE)
    y_ = net(x)

    #loss calculate
    loss = common.cal_loss(y_, y)
    loss.backward()

    # optimizer
    optimizer.step()

    # add the loss into writer
    losses.append(loss.item())
    if (step % PRINT_STEP) == 0:
        # print loss
        writer.add_scalar("training loss", loss.item(), step)
        print("Loss: ", np.mean(losses), "\nStep: ", step)

    if (step % VALIDATION_STEP) == 0:
        # validation
        accuracy, test_loss = validator.test(test_set, step)

    step = step + 1