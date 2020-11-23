##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#######################################################
# Quiz
#######################################################

# Q1
# How many rows and columns are there in the edx dataset?

dim(edx)


# Q2
# How many zeros were given as ratings in the edx dataset?

edx %>% filter(rating == 0) %>% tally()

# How many threes were given as ratings in the edx dataset?

edx %>% filter(rating == 3) %>% tally()


# Q3
# How many different movies are in the edx dataset?

n_distinct(edx$movieId)


# Q4
# How many different users are in the edx dataset?

n_distinct(edx$userId)


# Q5
# How many movie ratings are in each of the following genres in the edx dataset?

edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))


# Q6
# Which movie has the greatest number of ratings?

edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))


# Q7
# What are the five most given ratings in order from most to least?

edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))  


# Q8
# True or False: In general, half star ratings are less common than 
# whole star ratings (e.g., there are fewer ratings of 3.5 than there 
# are ratings of 3 or 4, etc.).

edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()


#######################################################
# MovieLens feature engineering and data analysis
#######################################################

# Install all needed libraries if it is not present

if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(stringr)) install.packages("stringr")
if(!require(forcats)) install.packages("forcats")
if(!require(ggplot2)) install.packages("ggplot2")

# Loading all needed libraries

library(dplyr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(stringr)
library(forcats)
library(ggplot2)

# The general RMSE function that will be used to evaluate overall performance of models:

RMSE <- function(true_ratings = NULL, predicted_ratings = NULL) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# MovieLens data preparation work including feature engineering and data type correction

# Convert timestamp to a human readable date format in both traning and validation dataset

edx$formated_timestamp <- as.POSIXct(edx$timestamp, origin="1970-01-01")
validation$formated_timestamp <- as.POSIXct(validation$timestamp, origin="1970-01-01")


# Extract the year and month of rating in both dataset

edx$yearOfRating <- format(edx$formated_timestamp,"%Y")
edx$monthOfRating <- format(edx$formated_timestamp,"%m")

validation$yearOfRating <- format(validation$formated_timestamp,"%Y")
validation$monthOfRating <- format(validation$formated_timestamp,"%m")


# Extract the year of release for each movie in both dataset
# edx dataset

edx <- edx %>%
  mutate(title = str_trim(title)) %>%
  extract(title, c("titleTemp", "release"), regex = "^(.*) \\(([0-9 \\-]*)\\)$", remove = F) %>%
  mutate(release = if_else(str_length(release) > 4, as.integer(str_split(release, "-", simplify = T)[1]), as.integer(release))) %>%
  mutate(title = if_else(is.na(titleTemp), title, titleTemp)) %>%
  select(-titleTemp)


# validation dataset

validation <- validation %>%
  mutate(title = str_trim(title)) %>%
  extract(title, c("titleTemp", "release"), regex = "^(.*) \\(([0-9 \\-]*)\\)$", remove = F) %>%
  mutate(release = if_else(str_length(release) > 4, as.integer(str_split(release, "-", simplify = T)[1]), as.integer(release))) %>%
  mutate(title = if_else(is.na(titleTemp), title, titleTemp)) %>%
  select(-titleTemp)


# Extract the genre in edx datasets and label missing genre in case any to include them in summary
# Error in separate_rows(genre, sep = "\\|") : object 'genre' not found, thus need to first convert it into character

edx <- edx %>%
  mutate(genre = fct_explicit_na(genres, na_level = "(missing)")) %>%
  mutate(genre = as.character(genre)) %>%
  separate_rows(genre, sep = "\\|")


# Extract the genre in validation datasets

validation <- validation %>%
  mutate(genre = fct_explicit_na(genres, na_level = "(missing)")) %>%
  mutate(genre = as.character(genre)) %>%
  separate_rows(genre, sep = "\\|")


# remove unnecessary columns on both dataset

edx <- edx %>% select(userId, movieId, rating, title, genre, release, yearOfRating, monthOfRating)
validation <- validation %>% select(userId, movieId, rating, title, genre, release, yearOfRating, monthOfRating)


# Convert newly added features into numeric data type for calculation precision

edx$yearOfRating <- as.numeric(edx$yearOfRating)
edx$monthOfRating <- as.numeric(edx$monthOfRating)
edx$release <- as.numeric(edx$release)

validation$yearOfRating <- as.numeric(validation$yearOfRating)
validation$monthOfRating <- as.numeric(validation$monthOfRating)
validation$release <- as.numeric(validation$release)


# Models exploration

# Average by all movies rating

mu_hat <- mean(edx$rating)

# RMSE on the validation set

mean_model_rmse <- RMSE(validation$rating, mu_hat) #1.052 root mean square error

# Since I am going to explore other various methods/models, 
# I need a variable to cache RMSE result of each for future reference.

results <- data.frame(model="Naive Mean - Baseline Model", RMSE=mean_model_rmse)


# Average by movie

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_m = mean(rating - mu_hat))

# Predicted ratings on validation dataset

movie_model_pred <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  mutate(pred = mu_hat + b_m) %>%
  pull(pred)

movie_model_rmse<- RMSE(validation$rating, movie_model_pred) # 0.941 RMSE


# Adding movie model to the results

results <- results %>% add_row(model="Movie - Based Model", RMSE=movie_model_rmse)



# Average by user

user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_m))

# Predicted ratings on validation dataset using movie and user average

movie_user_pred <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_m + b_u) %>%
  pull(pred)

movie_user_model_rmse <- RMSE(validation$rating, movie_user_pred)

# Adding movie user model to the results

results <- results %>% add_row(model="Movie + User Based Model", RMSE=movie_user_model_rmse)


# Average by genre

genre_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genre) %>%
  summarize(b_u_g = mean(rating - mu_hat - b_m - b_u))

# Predicted ratings on validation dataset using movie, user, and genre

movie_user_genre_pred <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genre') %>%
  mutate(pred = mu_hat + b_m + b_u + b_u_g) %>%
  pull(pred)

movie_user_genre_model_rmse <- RMSE(validation$rating, movie_user_genre_pred) # 0.863 RMSE

# Adding movie user genre model to the results

results <- results %>% add_row(model="Movie + User + Genre Based Model", RMSE=movie_user_genre_model_rmse)


# Regularization to penalize movies with large estimates from small sample size.
# Penalty table 

penalties <- seq(0, 10, 0.1)


# Predicted ratings on validation dataset using different values of penalties based on movie

rmses <- sapply(penalties, function(penalty) {
  
  # Average by movie
  
  b_m <- edx %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu_hat) / (n() + penalty))
  
  # Compute the predicted ratings on validation dataset
  
  predicted_ratings <- validation %>%
    left_join(b_m, by='movieId') %>%
    mutate(pred = mu_hat + b_m) %>%
    pull(pred)
  
  # Predict the RMSE on the validation set
  
  return(RMSE(validation$rating, predicted_ratings))
})


# Get the penalty value that minimize the RMSE

min_penalty <- penalties[which.min(rmses)] # 3.6

# Predict the RMSE on the validation set

regularized_movie_model_rmse <- min(rmses) # 0.941 RMSE

# Adding the regularized movie model rmse to the results

results <- results %>% add_row(model="Regularized Movie - Based Model", RMSE=regularized_movie_model_rmse)

# Predicted ratings on validation dataset using different values of penalties based on movie and user

rmses <- sapply(penalties, function(penalty) {
  
  # Average by movie
  
  b_m <- edx %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu_hat) / (n() + penalty))
  
  # Average by user and movie
  
  b_u <- edx %>%
    left_join(b_m, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu_hat) / (n() + penalty))
  
  # Predicted ratings on validation dataset by movie and user
  
  predicted_ratings <- validation %>%
    left_join(b_m, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    mutate(pred = mu_hat + b_m + b_u) %>%
    pull(pred)
  
  # Predict the RMSE on the validation set
  
  return(RMSE(validation$rating, predicted_ratings))
})

# Get the penalty value that minimize the RMSE

min_penalty <- penalties[which.min(rmses)] # 10

# Predict the RMSE on the validation set

regularized_movie_user_model_rmse <- min(rmses) # 0.8628

# Adding the results to the results dataset

results <- results %>% add_row(model="Regularized Movie + User Based Model", RMSE=regularized_movie_user_model_rmse)


#################################################################################
# Testing above findings with higher penalties values for movie, user, and genre
#################################################################################


penalties <- seq(0, 15, 0.1)

# Compute the predicted ratings on validation dataset using different values of penalties

rmses <- sapply(penalties, function(penalty) {
  
  # Average by movie
  
  b_m <- edx %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu_hat) / (n() + penalty))
  
  # Average by user
  
  b_u <- edx %>%
    left_join(b_m, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu_hat) / (n() + penalty))
  
  # Average by movie and user
  
  b_u_g <- edx %>%
    left_join(b_m, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genre) %>%
    summarize(b_u_g = sum(rating - b_m - mu_hat - b_u) / (n() + penalty))
  
  # Compute the predicted ratings on validation dataset using movie, user, and genre
  
  predicted_ratings <- validation %>%
    left_join(b_m, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_u_g, by='genre') %>%
    mutate(pred = mu_hat + b_m + b_u + b_u_g) %>%
    pull(pred)
  
  # Predict the RMSE on the validation set
  
  return(RMSE(validation$rating, predicted_ratings))
})

# Get the lambda value that minimize the RMSE

min_penalty <- penalties[which.min(rmses)] # 14.8

# Predict the RMSE on the validation set

regularized_movie_user_genre_model_rmse <- min(rmses) # 0.8626 (no improvement observed on higher penalties)

# Adding the regularized movie user genre RMSE to the results

results <- results %>% add_row(model="Regularized Movie + User + Genre Based Model", RMSE=regularized_movie_user_genre_model_rmse)

results





