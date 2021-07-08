library(dplyr)
library(glue)
library(magrittr)
library(readr)

# Load results into tibble

dir           <- '/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/10_AddRnaModel'
no_model_path <- glue('{dir}/decode-4-no-rna-model-MOD-ED.tsv')
# model_9_path  <- glue('{dir}/decode-2.tsv')
model_30_path  <- glue('{dir}/decode-6-rna-model-30-MOD-ED.tsv')

no_model_path %>%
  read_tsv(col_names=c('read_n', 'read', 'gt', 'pred_nm', 'ed_nm'), skip=1) %>%
  print() ->
ds_no_model

# model_9_path %>%
#   read_tsv(col_names=c('read', 'gt', 'pred_9', 'ed_9')) %>%
#   print() ->
# ds_model_9

model_30_path %>%
  read_tsv(col_names=c('read_n', 'read', 'gt', 'pred_30', 'ed_30'), skip=1) %>%
  print() ->
ds_model_30

ds_no_model %>%
  # inner_join(ds_model_9) %>%
  inner_join(ds_model_30) %>%
  print() %>%
  select('read', 'ed_nm', 'ed_30') %>%
  print() ->
ds_joined

# ds_joined %>%
#   filter(ed_9 < ed_nm) %>%
#   nrow() %>%
#   print() ->
# n_better_9

# ds_joined %>%
#   filter(ed_9 > ed_nm) %>%
#   nrow() %>%
#   print() ->
# n_worse_9

# ds_joined %>%
#   filter(ed_9 == ed_nm) %>%
#   nrow() %>%
#   print() ->
# n_same_9

ds_joined %>%
  filter(ed_30 < ed_nm) %>%
  nrow() %>%
  print() ->
n_better_30

ds_joined %>%
  filter(ed_30 > ed_nm) %>%
  nrow() %>%
  print() ->
n_worse_30

ds_joined %>%
  filter(ed_30 == ed_nm) %>%
  nrow() %>%
  print() ->
n_same_30

ds_joined %>%
  select(ed_nm) %>%
  pull() %>%
  mean() ->
mean_nm

ds_joined %>%
  select(ed_30) %>%
  pull() %>%
  mean() ->
mean_30





dir           <- '/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/8_TestGlobalDecoding/3_ComputeEditDistance'
global_path <- glue('{dir}/edit_distance_global_heart.tsv')
# model_9_path  <- glue('{dir}/decode-2.tsv')
  # <- glue('{dir}/decode-6-rna-model-30-MOD-ED.tsv')

global_path %>%
  read_tsv(col_names=c('read', 'gt', 'pred', 'ed')) %>%
  print() ->
ds_global

ds_global %>%
  select(ed) %>%
  pull() %>%
  mean() %>%
  print()


ds_no_model %>%
  select(ed_nm) %>%
  pull() %>%
  mean() %>%
  print()

ds_global %>%
  inner_join(ds_no_model) %>%
  print() %>%
  select(read, ed, ed_nm) %>%
  print() %>%
  filter(ed != ed_nm) %>%
  print()
