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

write_train_data <- function(data, path) {
  readr::write_csv(
    data,
    path = path,
    col_names = T,
    append = F
  )
}
read_train_data <- function(path) {
  readr::read_csv(
    file = path,
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

write_test_data <- function(data, path) {
  readr::write_csv(
    data,
    path = path,
    col_names = T,
    append = F
  )
}
read_test_data <- function(path) {
  readr::read_csv(
    file = path,
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
create_cv <- function(data, v = 5, seed = 1726) {
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

      # 行ごとのゆらぎ
      min_characters = min(characters),
      max_characters = max(characters),
      sd_characters  = sd(characters),
      min_katakana_ratio = min(katakanas / characters),
      max_katakana_ratio = max(katakanas / characters),
      sd_katakana_ratio  = sd(katakanas  / characters),
      min_kanji_ratio = min(kanjis / characters),
      max_kanji_ratio = max(kanjis / characters),
      sd_kanji_ratio  = sd(kanjis  / characters),
      min_touten_ratio = min(toutens / characters),
      max_touten_ratio = max(toutens / characters),
      sd_touten_ratio  = sd(toutens  / characters),
      min_touten = min(toutens),
      max_touten = max(toutens),
      sd_touten  = sd(toutens),

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

get_morphological_data <- function(data, path) {
  
  if(!file.exists(path)) {
    make_morphological_properties(data) %>%
      readr::write_csv(path = path, append = F, col_names = T)
  }

  readr::read_csv(
    file = path,
    col_types = cols(
      writing_id = col_integer(),
      pos = col_character(),
      morph = col_character()
    )
  )
}


get_top_morphes <- function(data) {

  data %>%

    # 読点と句点を除去
    dplyr::filter(
      !(morph %in% c("、", "。"))
    ) %>%

    dplyr::count(pos, morph, sort = T) %>%

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

add_morph_info <- function(target_data, morph_data) {

  # 上位の形態素を抽出
  df.top_morphes <- get_top_morphes(morph_data)

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

add_pos_ratio_before_period <- function(target_data, morph_data, period, prefix) {

  df.pos_ratios <- morph_data %>%

    # 読点の直前の形態素のみを抽出
    dplyr::group_by(writing_id) %>%
    dplyr::mutate(next_morph = lead(morph)) %>%
    dplyr::ungroup() %>%
    dplyr::filter(next_morph == period) %>%

    # 品詞ごとにカウント
    dplyr::count(writing_id, pos) %>%

    # 品詞ごとの比率を算出
    dplyr::group_by(writing_id) %>%
    dplyr::mutate(ratio = n / sum(n)) %>%
    dplyr::ungroup() %>%

    dplyr::mutate(
      pos = dplyr::case_when(
        pos == "名詞"     ~ "pos_01",
        pos == "動詞"     ~ "pos_02",
        pos == "形容詞"   ~ "pos_03",
        pos == "助詞"     ~ "pos_04",
        pos == "助動詞"   ~ "pos_05",
        pos == "副詞"     ~ "pos_06",
        pos == "接続詞"   ~ "pos_07",
        pos == "接頭詞"   ~ "pos_08",
        pos == "連体詞"   ~ "pos_09",
        pos == "感動詞"   ~ "pos_10",
        pos == "フィラー" ~ "pos_11",
        pos == "記号"     ~ "pos_12",
        T                 ~ "pos_99"
      )
    ) %>%
    dplyr::arrange(pos) %>%

    # long-form => wide-form
    dplyr::select(-n) %>%
    tidyr::pivot_wider(names_from = pos, names_prefix = prefix, values_from = ratio, values_fill = list(ratio = 0))

  dplyr::left_join(target_data, df.pos_ratios, by = "writing_id")
}


