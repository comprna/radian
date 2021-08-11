library(dplyr)
library(glue)
library(magrittr)
library(readr)

# Load results into tibble

dir           <- '/mnt/sda/rna-basecaller/experiments/with-rna-model/global/all_val'
no_model_path <- glue('{dir}/decode-4/decode-4.tsv')
model_9_path  <- glue('{dir}/decode-5/decode-5.tsv')
model_30_path <- glue('{dir}/decode-6/decode-6.tsv')

no_model_path %>%
  read_tsv(col_names=c('read', 'gt', 'pred_nm', 'ed_nm')) %>%
  print() ->
ds_no_model

model_9_path %>%
  read_tsv(col_names=c('read', 'gt', 'pred_9', 'ed_9')) %>%
  print() ->
ds_model_9

model_30_path %>%
  read_tsv(col_names=c('read', 'gt', 'pred_30', 'ed_30')) %>%
  print() ->
ds_model_30

ds_no_model %>%
  inner_join(ds_model_9) %>%
  inner_join(ds_model_30) %>%
  print() %>%
  select('read', 'ed_nm', 'ed_9', 'ed_30') %>%
  print(n=50) ->
ds_joined

ds_joined %>%
  filter(ed_9 < ed_nm) %>%
  nrow() %>%
  print() ->
n_better_9

ds_joined %>%
  filter(ed_9 > ed_nm) %>%
  nrow() %>%
  print() ->
n_worse_9

ds_joined %>%
  filter(ed_9 == ed_nm) %>%
  nrow() %>%
  print() ->
n_same_9

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
  select(ed_9) %>%
  pull() %>%
  mean() ->
mean_9

ds_joined %>%
  select(ed_30) %>%
  pull() %>%
  mean() ->
mean_30

mean_nm
mean_9
mean_30
