library(readr)
library(dplyr)
library(magrittr)

# Load the results from file

load_results <- function(results_file)
{
  results_file %>%
    read_csv(col_names = FALSE) %>%
    rename(expected  = X1,
           predicted = X2,
           ed        = X3)
}

greedy_file     <- "/mnt/sda/rna-basecaller/experiments/4_8_NNInputs/curriculum_learning/train-10/tests/train-10-test-greedy.sh.o18529352-clean"
beam_file       <- "/mnt/sda/rna-basecaller/experiments/4_8_NNInputs/curriculum_learning/train-10/tests/train-10-test-beam.sh.o18529381-clean"
model_file      <- "/mnt/sda/rna-basecaller/experiments/4_8_NNInputs/curriculum_learning/train-10/tests/train-10-test-rna-model.sh.o18529427-clean"
cond_model_file <- "/mnt/sda/rna-basecaller/experiments/4_8_NNInputs/curriculum_learning/train-10/tests/train-10-test-rna-cond-model.sh.o18530456-clean"

greedy     <- load_results(greedy_file)
beam       <- load_results(beam_file)
model      <- load_results(model_file)
cond_model <- load_results(cond_model_file)

# Combine all of the results into a single tibble for comparison

merged <- bind_cols(greedy, beam, model, cond_model)

# Rename columns to improve readability

merged %<>%
  rename(expected_greedy = expected...1,
         predicted_greedy = predicted...2,
         ed_greedy = ed...3,
         expected_beam = expected...4,
         predicted_beam = predicted...5,
         ed_beam = ed...6,
         expected_model = expected...7,
         predicted_model = predicted...8,
         ed_model = ed...9,
         expected_cond = expected...10,
         predicted_cond = predicted...11,
         ed_cond = ed...12)

# Check the expected sequences are the same to ensure merge is correct

merged %>%
  select(expected_greedy, expected_beam, expected_model, expected_cond) %>%
  mutate(match = expected_beam  == expected_greedy &&
                 expected_model == expected_greedy &&
                 expected_cond  == expected_greedy) %>%
  filter(match = FALSE) %>%
  print() # This should be an empty tibble

# No need to repeat expected columns, since they are the same

merged %<>%
  select(expected_greedy, predicted_greedy, ed_greedy, predicted_beam,
         ed_beam, predicted_model, ed_model, predicted_cond, ed_cond) %>%
  rename(expected = expected_greedy)

# Get the average edit distance for each decoder

merged %>%
  select(expected, predicted_greedy, ed_greedy, predicted_beam,
         ed_beam, predicted_model, ed_model, predicted_cond, ed_cond) %>%
  summarise_all(mean) ->
mean_eds

mean_eds

# Find instances where conditional model is better than beam

merged %>%
  select(expected, predicted_beam, ed_beam, predicted_cond, ed_cond) %>%
  filter(ed_cond < ed_beam) ->
ed_cond_better

ed_cond_better # 14,319 instances = 3.18%

# Find instances where conditional model is worse than beam

merged %>%
  select(expected, predicted_beam, ed_beam, predicted_cond, ed_cond) %>%
  filter(ed_cond > ed_beam) ->
ed_cond_worse

ed_cond_worse # 25,879 instances = 5.75%

# Find instances where conditional model is same as beam

merged %>%
  select(expected, predicted_beam, ed_beam, predicted_cond, ed_cond) %>%
  filter(ed_cond == ed_beam) ->
ed_cond_same

ed_cond_same # 409,772 instances = 91%

# ED Beam average = 0.289
# ED RNA conditional model average = 0.305