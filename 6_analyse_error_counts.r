library(dplyr)
library(glue)
library(magrittr)
library(readr)

# Load results into tibble

filepath <- 'error_counts_guppy_heart.tsv'

filepath %>%
  read_tsv() %>%
  print() ->
ds

# Analyse error %

ds %<>%
  mutate(pcent_id    = n_mat / gt_length * 100,
         tot_error   = n_sub + n_ins + n_del,
         tot_pcent   = tot_error / gt_length * 100,
         sub_pcent   = n_sub / gt_length * 100,
         ins_pcent   = n_ins / gt_length * 100,
         ins_pcent2  = n_ins / aln_length * 100,
         del_pcent   = n_del / gt_length * 100,
         hdel_pcent  = n_hdel / n_del * 100,
         ctsub_pcent = n_ctsub / n_sub * 100,
         cgsub_pcent = n_cgsub / n_sub * 100,
         casub_pcent = n_casub / n_sub * 100,
         gasub_pcent = n_gasub / n_sub * 100,
         gtsub_pcent = n_gtsub / n_sub * 100,
         atsub_pcent = n_atsub / n_sub * 100,
         )

print(mean(ds[['n_alignments']]))
print(mean(ds[['pcent_id']]))
print(mean(ds[['tot_pcent']]))
print(mean(ds[['sub_pcent']]))
print(mean(ds[['ins_pcent']]))
print(mean(ds[['ins_pcent2']]))
print(mean(ds[['del_pcent']]))
print(mean(ds[['hdel_pcent']], na.rm=TRUE))
print(mean(ds[['ctsub_pcent']], na.rm=TRUE))
print(mean(ds[['cgsub_pcent']], na.rm=TRUE))
print(mean(ds[['casub_pcent']], na.rm=TRUE))
print(mean(ds[['gasub_pcent']], na.rm=TRUE))
print(mean(ds[['gtsub_pcent']], na.rm=TRUE))
print(mean(ds[['atsub_pcent']], na.rm=TRUE))



