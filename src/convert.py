# import dataset
import config
import pandas as pd
import numpy as np

paths = []
targets = []
f = open(config.IMAGE_FILE_TXT)
for l in f:
    p, t = l.split(" ")
    p = config.IMAGE_FOLDER + p
    paths.append(p)
    t = (t.replace('\n', ""))
    t = float(t) * 3.14 / 180.0
    targets.append(t)
f.close()

d = {'path': paths, 'target': targets}

df = pd.DataFrame(d, columns=['path', 'target'])

df.to_csv(config.IMAGE_FOLDER+"data.csv", index=False)
print("DataFrame saved to the folder")
df = pd.read_csv(config.IMAGE_FOLDER+"data.csv")
print(df.tail())

