# train_f: 0.9991433, test_f: 0.9943168 めっちゃ精度が出てるんだけど・・・


library(tidyverse)
library(tidymodels)
library(furrr)
library(lightgbm)

source("models/LightGBM_01/functions.R", encoding = "utf-8")


# Load Train Data -------------------------------------------------------

# Train Data
# system.time({
#   df.train.raw <- load_train_data("data/input/train.csv")
# 
#   df.train <- df.train.raw %>%
# 
#     # 行単位の特徴量を追加
#     add_line_info(df.train.raw) %>%
# 
#     # 形態素関連の特徴量を追加
#     add_morph_info()
# })
#write_train_data(df.train)
df.train <- read_train_data()

# Train - CV
df.cv <- create_cv(df.train)


# Recipe ------------------------------------------------------------------

rec <- recipes::recipe(author ~ ., df.train) %>%

  recipes::step_mutate(
    kutens_per_morphes  = line_count / total_morphes,
    toutens_per_morphes = toutens    / total_morphes
  ) %>%

  recipes::step_rm(
    writing_id,
    body,

    line_count,
    characters,
    kanjis,
    katakanas,
    toutens,
    characters_per_line,
    kanjis_per_line,
    katakanas_per_line,
    toutens_per_line,
    katakanas_per_characters,
    kanjis_per_characters,
    toutens_per_characters,
    kutens_per_characters,
    total_morphes
  )

recipes::prep(rec) %>%
  recipes::juice() %>%
  dplyr::select(
    -dplyr::starts_with("morph"),
    morph_1,
    morph_2,
    morph_3
  ) %>%
  summary()


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- tibble(
  max_depth = 10,
  num_leaves = 35,
  min_data_in_leaf = 25,

  feature_fraction = 0.5,
  bagging_freq = 5,
  bagging_fraction = 0.95,

  lambda_l1 = 0.0,
  lambda_l2 = 0.0
) 
# %>%
#   tidyr::crossing(
#     max_depth = seq(10, 12, 1),
#     num_leaves = seq(35, 37, 1),
#     min_data_in_leaf = seq(24, 26, 1)
#   )
df.grid.params


# Parametr Fitting --------------------------------------------------------

# 並列処理
future::plan(future::multisession(workers = 8))

system.time({

  df.results <- purrr::pmap_dfr(df.grid.params, function(
    max_depth,
    num_leaves,
    min_data_in_leaf,

    feature_fraction,
    bagging_freq,
    bagging_fraction,

    lambda_l1,
    lambda_l2
  ) {

    hyper_params <- list(
      max_depth        = max_depth,
      num_leaves       = num_leaves,
      min_data_in_leaf = min_data_in_leaf,
      feature_fraction = feature_fraction,
      bagging_freq     = bagging_freq,
      bagging_fraction = bagging_fraction,

      lambda_l1 = lambda_l1,
      lambda_l2 = lambda_l2
    )

    furrr::future_map_dfr(df.cv$splits, function(split, recipe, hyper_params) {

      # 前処理済データの作成
      lst.train_valid_test <- recipe %>%
        {
          recipe <- (.)

          # train data
          df.train <- recipes::prep(recipe) %>%
            recipes::bake(rsample::training(split))
          x.train <- df.train %>%
            dplyr::select(-author) %>%
            as.matrix()
          y.train <- df.train$author

          # for early_stopping
          train_valid_split <- rsample::initial_split(df.train, prop = 4/5, strata = "author")
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
          df.test  <- recipes::prep(recipe) %>%
            recipes::bake(rsample::testing(split))
          x.test <- df.test %>%
            dplyr::select(-author) %>%
            as.matrix()
          y.test <- df.test$author


          list(
            ## model 学習用
            train.dtrain = dtrain,
            train.dvalid = dvalid,

            # MAE 算出用: train
            x.train = x.train,
            y.train = y.train,

            ## MAE 算出用: test
            x.test = x.test,
            y.test = y.test
          )
        }

      # 学習
      model.fitted <- lightgbm::lgb.train(

        # 学習パラメータの指定
        params = list(
          boosting  = "gbdt",
          objective = "binary",
          metric    = "binary_logloss",

          # user defined
          max_depth        = hyper_params$max_depth,
          num_leaves       = hyper_params$num_leaves,
          min_data_in_leaf = hyper_params$min_data_in_leaf,
          feature_fraction = hyper_params$feature_fraction,
          bagging_freq     = hyper_params$bagging_freq,
          bagging_fraction = hyper_params$bagging_fraction,
          lambda_l1        = hyper_params$lambda_l1,
          lambda_l2        = hyper_params$lambda_l2,

          seed = 1234
        ),

        # 学習＆検証データ
        data   = lst.train_valid_test$train.dtrain,
        valids = list(valid = lst.train_valid_test$train.dvalid),

        # 木の数など
        learning_rate = 0.1,
        nrounds = 20000,
        early_stopping_rounds = 200,
        verbose = -1
      )

      # F の算出
      train_f <- tibble::tibble(
        actual = lst.train_valid_test$y.train,
        pred   = (predict(model.fitted, lst.train_valid_test$x.train) > 0.5)
      ) %>%
        yardstick::f_meas(truth = factor(actual), estimate = factor(pred)) %>%
        .$.estimate
      test_f <- tibble::tibble(
        actual = lst.train_valid_test$y.test,
        pred   = (predict(model.fitted, lst.train_valid_test$x.test) > 0.5)
      ) %>%
        yardstick::f_meas(truth = factor(actual), estimate = factor(pred)) %>%
        .$.estimate

      tibble::tibble(
        train_f = train_f,
        test_f  = test_f
      )
    }, recipe = rec, hyper_params = hyper_params, .options = furrr::future_options(seed = 5963L)) %>%

      # CV 分割全体の平均値を評価スコアとする
      dplyr::summarise_all(mean)
  }) %>%

    # 評価結果とパラメータを結合
    dplyr::bind_cols(df.grid.params, .) %>%

    # 評価スコアの順にソート(昇順)
    dplyr::arrange(test_f)
})
