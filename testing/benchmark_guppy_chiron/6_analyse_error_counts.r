library(dplyr)
library(glue)
library(magrittr)
library(readr)

# Load results into tibble

filepath <- 'error_counts_tcn_hek293.tsv'

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

print(glue("Mean number of alignments: {mean(ds[['n_alignments']])}"))
print(glue("Mean % identity: {mean(ds[['pcent_id']])}"))
print(glue("Mean total error %: {mean(ds[['tot_pcent']])}"))
print(glue("Mean sub %: {mean(ds[['sub_pcent']])}"))
print(glue("Mean ins % (denom=GT length): {mean(ds[['ins_pcent']])}"))
print(glue("Mean ins % (denom=Aln length): {mean(ds[['ins_pcent2']])}"))
print(glue("Mean del %: {mean(ds[['del_pcent']])}"))
print(glue("Mean homopolymer del % (as % of del): {mean(ds[['hdel_pcent']], na.rm=TRUE)}"))
print(glue("Mean CT sub % (as % of sub): {mean(ds[['ctsub_pcent']], na.rm=TRUE)}"))
print(glue("Mean CG sub % (as % of sub): {mean(ds[['cgsub_pcent']], na.rm=TRUE)}"))
print(glue("Mean CA sub % (as % of sub): {mean(ds[['casub_pcent']], na.rm=TRUE)}"))
print(glue("Mean GA sub % (as % of sub): {mean(ds[['gasub_pcent']], na.rm=TRUE)}"))
print(glue("Mean GT sub % (as % of sub): {mean(ds[['gtsub_pcent']], na.rm=TRUE)}"))
print(glue("Mean AT sub % (as % of sub): {mean(ds[['atsub_pcent']], na.rm=TRUE)}"))
