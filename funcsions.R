library(tidyverse)
library(furrr)
library(RMeCab)


# Train
load_train_data <- function(path) {

  readr::read_csv(
    file = path,
    col_types = cols(
      writing_id = col_integer(),
      body = col_character(),
      author = col_logical()
    )
  ) %>%

    # 不要項目の除去: 空白/改行
    dplyr::mutate(
      body = stringr::str_remove_all(body, pattern = "\\s")
    )
}

write_train_data <- function(data) {
  readr::write_csv(
    data,
    path = "models/LightGBM_01/data/input/train_data.csv",
    col_names = T,
    append = F
  )
}
read_train_data <- function() {
  readr::read_csv(
    file = "models/LightGBM_01/data/input/train_data.csv",
    col_types = cols(
      .default = col_double(),
      writing_id = col_integer(),
      body = col_character(),
      author = col_logical(),
      line_count = col_integer(),
      characters = col_integer(),
      kanjis = col_integer(),
      katakanas = col_integer(),
      toutens = col_integer()
    )
  )
}

# Test
load_test_data <- function(path) {

  readr::read_csv(
    file = path,
    col_types = cols(
      writing_id = col_integer(),
      body = col_character()
    )
  ) %>%

    # 不要項目の除去: 空白/改行
    dplyr::mutate(
      body = stringr::str_remove_all(body, pattern = "\\s")
    )
}

write_test_data <- function(data) {
  readr::write_csv(
    data,
    path = "models/LightGBM_01/data/input/test_data.csv",
    col_names = T,
    append = F
  )
}
read_test_data <- function() {
  readr::read_csv(
    file = "models/LightGBM_01/data/input/test_data.csv",
    col_types = cols(
      .default = col_double(),
      writing_id = col_integer(),
      body = col_character(),
      line_count = col_integer(),
      characters = col_integer(),
      kanjis = col_integer(),
      katakanas = col_integer(),
      toutens = col_integer()
    )
  )
}


# Cross Validation
create_cv <- function(data, v = 5, seed = 1234) {
  set.seed(seed)
  
  rsample::vfold_cv(data, v = v, strata = "author")
}


# 行・文字 --------------------------------------------------------------------

# 行ごとの統計量を生成
make_line_properties <- function(data) {

  data %>%

    dplyr::mutate(
      lines = purrr::map(body, function(body) {
        # 行ごとに分離
        stringr::str_split(body, pattern = "。") %>%
          purrr::flatten_chr() %>%
          tibble::enframe(name = "line_no", value = "text") %>%

          # 空行を除去
          dplyr::filter(text != "")
      })
    ) %>%

    dplyr::select(-body) %>%
    tidyr::unnest(lines) %>%

    dplyr::mutate(
      # 文字数
      # 句点の分だけ +1 する
      characters = stringr::str_length(text) + 1,

      # 文字数: ひらがな
      hiraganas = stringr::str_count(text, pattern = "\\p{Hiragana}"),

      # 文字数: カタカナ
      katakanas = stringr::str_count(text, pattern = "\\p{Katakana}"),

      # 文字数: 漢字
      kanjis = stringr::str_count(text, pattern = "\\p{Han}"),

      # 読点(とうてん)の数
      toutens = stringr::str_count(text, pattern = "、")
    )
}

# 行・文字単位の統計量を追加
add_line_info <- function(target_data, rawdata) {

  df.line_info <- rawdata %>%

    # 行単位の統計量を生成
    make_line_properties() %>%

    # 作品単位に集計
    dplyr::group_by(writing_id) %>%
    dplyr::summarise(
      # 行全体の統計量
      line_count = n(),
      characters = sum(characters),
      kanjis     = sum(kanjis),
      katakanas  = sum(katakanas),
      toutens    = sum(toutens),

      # 行単位の統計量
      characters_per_line = characters / line_count,
      kanjis_per_line     = kanjis     / line_count,
      katakanas_per_line  = katakanas  / line_count,
      toutens_per_line    = toutens    / line_count,

      # 文字あたりの統計量
      katakanas_per_characters = katakanas  / characters,
      kanjis_per_characters    = kanjis     / characters,
      toutens_per_characters   = toutens    / characters,
      kutens_per_characters    = line_count / characters
    ) %>%
    dplyr::ungroup()

  # 行単位の情報を追加
  target_data %>%
    dplyr::left_join(df.line_info, by = "writing_id")
}

# 形態素 -----------------------------------------------------------------

make_morphological_properties <- function(data, mypref = 1) {

#  # 並列処理
#  future::plan(future::multisession(workers = 1))

  data %>%

    # データを workers 個の塊に分割
    dplyr::mutate(chunk_id = (writing_id %% 1)) %>%
    dplyr::group_by(chunk_id) %>%
    tidyr::nest() %>%
    .$data %>%

    # 並列実行
    purrr::map_dfr(function(chunk) {
      chunk %>%

        # 形態素解析の実施
        dplyr::mutate(
          mecab = purrr::map(body, function(text, mypref) {
            text %>%
              RMeCab::RMeCabC(mypref = mypref) %>%
              purrr::flatten_chr() %>%
              tibble::enframe(name = "pos", value = "morph")
          }, mypref = mypref)
        ) %>%
        
        dplyr::select(-body) %>%
        tidyr::unnest(mecab)
    })
}
write_morphological_data <- function(data, path) {
  readr::write_csv(data, path = path, append = F, col_names = T)
}
load_morphological_data <- function(path) {
  readr::read_csv(
    file = path,
    col_types = cols(
      pos = col_character(),
      morph = col_character(),
      n = col_integer()
    )
  )
}
# make_morphological_properties(df.train.raw) %>%
#   write_morphological_data(path = "models/LightGBM_01/data/input/morphological_data.csv")
# load_morphological_data("models/LightGBM_01/data/input/morphological_data.csv")

get_top_morphes <- function(data) {

  data %>%

    # 読点と句点を除去
    dplyr::filter(
      !(morph %in% c("、", "。"))
    ) %>%

    dplyr::mutate(
      rank = dplyr::row_number(desc(n)),
      rank_ratio = rank / max(rank),
      cumsum_n_ratio = cumsum(n) / sum(n)
    ) %>%

    dplyr::filter(
      rank <= 10000
    ) %>%

    dplyr::select(
      morph_id = rank,
      morph
    )
}
#get_top_morphes(df.morphes)

add_morph_info <- function(target_data, morph_data = NULL) {

  # 形態素解析の実行
  df.morphes <- morph_data
  if(is.null(df.morphes)) {
    df.morphes <- load_morphological_data("models/LightGBM_01/data/input/morphological_data.csv")
  }

  # 上位の形態素を抽出
  df.top_morphes <- get_top_morphes(df.morphes)

  df.morph_ratios <- target_data %>%

    # 形態素解析を実施
    make_morphological_properties() %>%

    # 作品ごとに各形態素の出現回数を算出
    dplyr::count(writing_id, morph) %>%

    # 読点と句点を対象外
    dplyr::filter(!(morph %in% c("、", "。"))) %>%

    # 各形態素の全体に占める割合を算出
    dplyr::group_by(writing_id) %>%
    dplyr::mutate(
      total_morphes = sum(n),
      ratio = n / total_morphes
    ) %>%
    dplyr::ungroup() %>%

    # 上位の形態素のみに制限
    dplyr::inner_join(df.top_morphes, by = "morph") %>%

    # long-form => wide-form
    dplyr::select(writing_id, total_morphes, morph_id, ratio) %>%
    dplyr::arrange(morph_id) %>%
    tidyr::pivot_wider(names_from = morph_id, names_prefix = "morph_", values_from = ratio, values_fill = list(ratio = 0))  

  target_data %>%
    dplyr::left_join(df.morph_ratios, by = "writing_id")
}



