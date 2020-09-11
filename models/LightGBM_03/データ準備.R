library(tidyverse)

source("models/LightGBM_01/functions.R", encoding = "utf-8")

# 形態素解析データの作成
df.morphes <- read_train_data() %>%
  dplyr::select(writing_id, body) %>%
  get_morphological_data("models/LightGBM_03/data/input/morphological_data.csv")

