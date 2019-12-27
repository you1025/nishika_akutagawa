library(tidyverse)
library(tidymodels)
library(lightgbm)
library(furrr)


# データ作成 -------------------------------------------------------------------

# test data
# load_test_data("data/input/test.csv") %>%
#   {
#     data <- (.)
# 
#     df.morphes <- load_morphological_data("models/LightGBM_01/data/input/morphological_data.csv")
# 
#     data %>%
# 
#       # 行・文字単位の特徴量を追加
#       add_line_info(df.test.raw) %>%
# 
#       # 形態素関連の特徴量を追加
#       add_morph_info(df.morphes) %>%
# 
#       # ファイルへの書き出し
#       write_test_data()    
# }

# 前処理済データの作成
lst.train_valid_test <- rec %>%
  {
    recipe <- (.) %>%
      recipes::prep()
    
    # train data
    df.train <- read_train_data() %>%
      # recipe の適用
      recipes::bake(object = recipe, new_data = .)
    x.train <- df.train %>%
      dplyr::select(-author) %>%
      as.matrix()
    y.train <- df.train$author
    
    # for early_stopping
    train_valid_split <- rsample::initial_split(df.train, prop = 8/9, strata = "author")
    x.train.train <- rsample::training(train_valid_split) %>%
      dplyr::select(-author) %>%
      as.matrix()
    y.train.train <- rsample::training(train_valid_split)$author
    x.train.valid <- rsample::testing(train_valid_split) %>%
      dplyr::select(-author) %>%
      as.matrix()
    y.train.valid <- rsample::testing(train_valid_split)$author
    
    # for LightGBM Dataset
    dtrain <- lightgbm::lgb.Dataset(
      data  = x.train.train,
      label = y.train.train
    )
    dvalid <- lightgbm::lgb.Dataset(
      data  = x.train.valid,
      label = y.train.valid,
      reference = dtrain
    )
    
    
    # test data
    df.test.raw <- load_test_data("data/input/test.csv")
    x.test <- read_test_data() %>%
      # recipe の適用
      recipes::bake(object = recipe, new_data = .) %>%
      # 検証データに足りない項目(morph_*)を追加
      {
        df <- (.)

        # 訓練データに含まれていて検証データに含まれないカラムの一覧を取得
        diffcols <- setdiff(colnames(x.train), colnames(df))

        # 検証用データの作成
        # 検証データに含まれない項目を追加して 0 で埋める
        purrr::reduce(diffcols, function(df, colname) {
          df[, colname] <- 0
          df
        }, .init = df) %>%

          # 項目の並び順を訓練データと合わせる
          dplyr::select(colnames(x.train)) %>%

          as.matrix()
      }
    
    
    list(
      ## model 学習用
      train.dtrain = dtrain,
      train.dvalid = dvalid,
      
      ## 提出用: test
      writing_id = df.test.raw$writing_id,
      x.test = x.test
    )
  }






df.grid.params <- tibble(
  max_depth = c(10),
  num_leaves = c(35),
  min_data_in_leaf = 25,

  feature_fraction = 0.5,
  bagging_freq = 5,
  bagging_fraction = 0.95,

  lambda_l1 = 0.0,
  lambda_l2 = 0.0
) %>%

  # random seed averaging
  tidyr::crossing(
    seed = sample(1:10000, size = 10, replace = F)
#    seed = 1234
  )

df.morphes <- load_morphological_data("models/LightGBM_01/data/input/morphological_data.csv")


# 並列処理
future::plan(future::multisession(workers = 8))

system.time({

  furrr::future_pmap_dfr(df.grid.params, function(
    max_depth,
    num_leaves,
    min_data_in_leaf,

    feature_fraction,
    bagging_freq,
    bagging_fraction,

    lambda_l1,
    lambda_l2,

    seed,

    recipe,
    morph_data,
    train_valid_test_list
  ) {

    # 学習パラメータの設定
    model.params <- list(
      boosting  = "gbdt",
      objective = "binary",
      metric    = "binary_logloss",

      # user defined
      max_depth        = max_depth,
      num_leaves       = num_leaves,
      min_data_in_leaf = min_data_in_leaf,
      feature_fraction = feature_fraction,
      bagging_freq     = bagging_freq,
      bagging_fraction = bagging_fraction,
      lambda_l1        = lambda_l1,
      lambda_l2        = lambda_l2,

      seed = seed
    )

    # 学習
    model.fitted <- lightgbm::lgb.train(

      # 学習パラメータの指定
      params = model.params,

      # 学習＆検証データ
      data   = train_valid_test_list$train.dtrain,
      valids = list(valid = train_valid_test_list$train.dvalid),

      # 木の数など
      learning_rate = 0.1,
      nrounds = 20000,
      early_stopping_rounds = 100,
      verbose = 1
    )

    # 予測結果
    predicted_probs <- predict(model.fitted, train_valid_test_list$x.test)
    tibble(
      writing_id = train_valid_test_list$writing_id,
      prob = predicted_probs
    )

  }, recipe = rec, morph_data = df.morphes, train_valid_test_list = lst.train_valid_test) %>%

    # 単一モデル内での Blending
    dplyr::group_by(writing_id) %>%
    dplyr::summarise(prob = mean(prob)) %>%

    dplyr::mutate(author = (prob > 0.5) %>% as.integer()) %>%
    dplyr::select(writing_id, author) %>%

    # ファイルに出力
    {
      df.submit <- (.)

      # ファイル名
      filename <- stringr::str_c(
        "LightGBM",
        lubridate::now(tz = "Asia/Tokyo") %>% format("%Y%m%dT%H%M%S"),
        sep = "_"
      ) %>%
        stringr::str_c("csv", sep = ".")

      # 出力ファイルパス
      filepath <- stringr::str_c("models/LightGBM_01/data/output", filename, sep = "/")

      # 書き出し
      readr::write_csv(df.submit, filepath, col_names = T)
    }
})

