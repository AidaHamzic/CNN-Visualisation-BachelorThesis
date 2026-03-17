import pandas as pd

df = pd.read_csv("outputs/vgg16_places365_manifest.csv")


print( df["predicted_label"].value_counts())

