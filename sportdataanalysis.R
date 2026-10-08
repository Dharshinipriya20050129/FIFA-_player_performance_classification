# ============================================================
# FIFA PLAYER PERFORMANCE CLASSIFICATION
# Decision Tree vs Random Forest
# ============================================================


# ------------------------------------------------------------
# 1. LOAD LIBRARIES
# ------------------------------------------------------------

library(ggplot2)
library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(tidyr)
library(scales)


# ------------------------------------------------------------
# 2. CHOOSE DATASET
# ------------------------------------------------------------

fifa_file <- file.choose()

fifa <- read.csv(
  fifa_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)


# ------------------------------------------------------------
# 3. SET R FILE FOLDER FOR GRAPH OUTPUT
# ------------------------------------------------------------

if (requireNamespace("rstudioapi", quietly = TRUE)) {
  
  r_file <- rstudioapi::getActiveDocumentContext()$path
  
  if (nzchar(r_file)) {
    setwd(dirname(r_file))
  }
  
}


# ------------------------------------------------------------
# 4. CHECK DATASET
# ------------------------------------------------------------

dim(fifa)

str(fifa)

summary(fifa)


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

missing_values <- colSums(is.na(fifa))

missing_table <- data.frame(
  Column = names(missing_values),
  Missing = as.numeric(missing_values)
)

missing_table <- missing_table %>%
  filter(Missing > 0) %>%
  arrange(desc(Missing))

print(missing_table)


# ============================================================
# 6. CHECK DUPLICATE RECORDS
# ============================================================

duplicates <- sum(duplicated(fifa))

cat(
  "Number of duplicate records:",
  duplicates,
  "\n"
)

fifa <- fifa %>%
  distinct()


# ============================================================
# 7. DATA CLEANING
# ============================================================

fifa$Age <- as.numeric(fifa$Age)

fifa$Overall <- as.numeric(fifa$Overall)

fifa$Potential <- as.numeric(fifa$Potential)


numeric_columns <- names(
  fifa
)[
  sapply(
    fifa,
    is.numeric
  )
]


for (col in numeric_columns) {
  
  fifa[[col]][is.na(fifa[[col]])] <-
    median(
      fifa[[col]],
      na.rm = TRUE
    )
  
}


# ============================================================
# 8. OVERALL RATING DISTRIBUTION
# ============================================================

overall_plot <- ggplot(
  fifa,
  aes(
    x = Overall
  )
) +
  geom_histogram(
    binwidth = 2,
    fill = "steelblue",
    color = "black"
  ) +
  labs(
    title = "Distribution of FIFA Player Overall Rating",
    x = "Overall Rating",
    y = "Number of Players"
  ) +
  theme_minimal()

print(overall_plot)

ggsave(
  "01_overall_distribution.png",
  overall_plot,
  width = 8,
  height = 5
)


# ============================================================
# 9. AGE DISTRIBUTION
# ============================================================

age_plot <- ggplot(
  fifa,
  aes(
    x = Age
  )
) +
  geom_histogram(
    binwidth = 1,
    fill = "orange",
    color = "black"
  ) +
  labs(
    title = "Age Distribution of FIFA Players",
    x = "Age",
    y = "Number of Players"
  ) +
  theme_minimal()

print(age_plot)

ggsave(
  "02_age_distribution.png",
  age_plot,
  width = 8,
  height = 5
)


# ============================================================
# 10. AGE VS OVERALL
# ============================================================

age_overall_plot <- ggplot(
  fifa,
  aes(
    x = Age,
    y = Overall
  )
) +
  geom_point(
    alpha = 0.3
  ) +
  geom_smooth(
    method = "lm",
    se = FALSE
  ) +
  labs(
    title = "Age vs Overall Rating",
    x = "Age",
    y = "Overall Rating"
  ) +
  theme_minimal()

print(age_overall_plot)

ggsave(
  "03_age_vs_overall.png",
  age_overall_plot,
  width = 8,
  height = 5
)


# ============================================================
# 11. TOP 10 NATIONALITIES
# ============================================================

top_nationalities <- fifa %>%
  count(
    Nationality,
    sort = TRUE
  ) %>%
  head(10)


nationality_plot <- ggplot(
  top_nationalities,
  aes(
    x = reorder(
      Nationality,
      n
    ),
    y = n
  )
) +
  geom_col(
    fill = "steelblue"
  ) +
  coord_flip() +
  labs(
    title = "Top 10 Nationalities of FIFA Players",
    x = "Nationality",
    y = "Number of Players"
  ) +
  theme_minimal()

print(nationality_plot)

ggsave(
  "04_top_nationalities.png",
  nationality_plot,
  width = 8,
  height = 5
)


# ============================================================
# 12. TOP 10 CLUBS
# ============================================================

top_clubs <- fifa %>%
  filter(
    !is.na(Club)
  ) %>%
  count(
    Club,
    sort = TRUE
  ) %>%
  head(10)


club_plot <- ggplot(
  top_clubs,
  aes(
    x = reorder(
      Club,
      n
    ),
    y = n
  )
) +
  geom_col(
    fill = "darkgreen"
  ) +
  coord_flip() +
  labs(
    title = "Top 10 Clubs by Number of Players",
    x = "Club",
    y = "Number of Players"
  ) +
  theme_minimal()

print(club_plot)

ggsave(
  "05_top_clubs.png",
  club_plot,
  width = 8,
  height = 5
)


# ============================================================
# 13. CREATE CLASSIFICATION TARGET
# ============================================================

# Overall >= 80 = High Rated
# Overall < 80  = Not High Rated


fifa$Performance <- ifelse(
  fifa$Overall >= 80,
  "High Rated",
  "Not High Rated"
)


fifa$Performance <- as.factor(
  fifa$Performance
)


print(
  table(
    fifa$Performance
  )
)


# ============================================================
# 14. PERFORMANCE CLASS DISTRIBUTION
# ============================================================

performance_count <- fifa %>%
  count(
    Performance
  )


performance_plot <- ggplot(
  performance_count,
  aes(
    x = Performance,
    y = n
  )
) +
  geom_col(
    fill = "purple"
  ) +
  labs(
    title = "FIFA Player Performance Classes",
    x = "Performance Category",
    y = "Number of Players"
  ) +
  theme_minimal()

print(performance_plot)

ggsave(
  "06_performance_classes.png",
  performance_plot,
  width = 8,
  height = 5
)


# ============================================================
# 15. SELECT MACHINE LEARNING FEATURES
# ============================================================

model_data <- fifa %>%
  select(
    Performance,
    Age,
    Potential,
    `International Reputation`,
    `Weak Foot`,
    `Skill Moves`,
    Crossing,
    Finishing,
    HeadingAccuracy,
    ShortPassing,
    Volleys,
    Dribbling,
    Curve,
    FKAccuracy,
    LongPassing,
    BallControl,
    Acceleration,
    SprintSpeed,
    Agility,
    Reactions,
    Balance,
    ShotPower,
    Jumping,
    Stamina,
    Strength,
    LongShots,
    Aggression,
    Interceptions,
    Positioning,
    Vision,
    Penalties,
    Composure,
    Marking,
    StandingTackle,
    SlidingTackle
  )


# Rename columns

names(model_data) <- c(
  "Performance",
  "Age",
  "Potential",
  "International_Reputation",
  "Weak_Foot",
  "Skill_Moves",
  "Crossing",
  "Finishing",
  "HeadingAccuracy",
  "ShortPassing",
  "Volleys",
  "Dribbling",
  "Curve",
  "FKAccuracy",
  "LongPassing",
  "BallControl",
  "Acceleration",
  "SprintSpeed",
  "Agility",
  "Reactions",
  "Balance",
  "ShotPower",
  "Jumping",
  "Stamina",
  "Strength",
  "LongShots",
  "Aggression",
  "Interceptions",
  "Positioning",
  "Vision",
  "Penalties",
  "Composure",
  "Marking",
  "StandingTackle",
  "SlidingTackle"
)


model_data <- na.omit(
  model_data
)


dim(model_data)

names(model_data)

summary(model_data)


# ============================================================
# 16. TRAIN TEST SPLIT
# ============================================================

set.seed(123)


train_index <- createDataPartition(
  model_data$Performance,
  p = 0.80,
  list = FALSE
)


train_data <- model_data[
  train_index,
]


test_data <- model_data[
  -train_index,
]


cat(
  "Training records:",
  nrow(train_data),
  "\n"
)


cat(
  "Testing records:",
  nrow(test_data),
  "\n"
)


# ============================================================
# 17. DECISION TREE MODEL
# ============================================================

decision_tree <- rpart(
  Performance ~ .,
  data = train_data,
  method = "class",
  control = rpart.control(
    cp = 0.01,
    maxdepth = 5
  )
)


print(decision_tree)


# Display tree

rpart.plot(
  decision_tree,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Decision Tree - FIFA Player Classification"
)


# Save tree

png(
  "07_decision_tree.png",
  width = 1000,
  height = 800
)


rpart.plot(
  decision_tree,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Decision Tree - FIFA Player Classification"
)


dev.off()


# ============================================================
# 18. DECISION TREE PREDICTION
# ============================================================

tree_prediction <- predict(
  decision_tree,
  test_data,
  type = "class"
)


tree_cm <- confusionMatrix(
  tree_prediction,
  test_data$Performance,
  positive = "High Rated"
)


print(tree_cm)


# ============================================================
# 19. RANDOM FOREST MODEL
# ============================================================

set.seed(123)


random_forest <- randomForest(
  Performance ~ .,
  data = train_data,
  ntree = 100,
  importance = TRUE
)


print(random_forest)


# ============================================================
# 20. RANDOM FOREST PREDICTION
# ============================================================

rf_prediction <- predict(
  random_forest,
  test_data
)


rf_cm <- confusionMatrix(
  rf_prediction,
  test_data$Performance,
  positive = "High Rated"
)


print(rf_cm)


# ============================================================
# 21. MODEL PERFORMANCE
# ============================================================

tree_accuracy <- as.numeric(
  tree_cm$overall[
    "Accuracy"
  ]
)


tree_precision <- as.numeric(
  tree_cm$byClass[
    "Precision"
  ]
)


tree_recall <- as.numeric(
  tree_cm$byClass[
    "Recall"
  ]
)


tree_f1 <- as.numeric(
  tree_cm$byClass[
    "F1"
  ]
)


rf_accuracy <- as.numeric(
  rf_cm$overall[
    "Accuracy"
  ]
)


rf_precision <- as.numeric(
  rf_cm$byClass[
    "Precision"
  ]
)


rf_recall <- as.numeric(
  rf_cm$byClass[
    "Recall"
  ]
)


rf_f1 <- as.numeric(
  rf_cm$byClass[
    "F1"
  ]
)


comparison <- data.frame(
  Algorithm = c(
    "Decision Tree",
    "Random Forest"
  ),
  Accuracy = c(
    tree_accuracy,
    rf_accuracy
  ),
  Precision = c(
    tree_precision,
    rf_precision
  ),
  Recall = c(
    tree_recall,
    rf_recall
  ),
  F1_Score = c(
    tree_f1,
    rf_f1
  )
)


print(comparison)


# ============================================================
# 22. ACCURACY COMPARISON
# ============================================================

accuracy_plot <- ggplot(
  comparison,
  aes(
    x = Algorithm,
    y = Accuracy
  )
) +
  geom_col(
    fill = "steelblue"
  ) +
  geom_text(
    aes(
      label = paste0(
        round(
          Accuracy * 100,
          2
        ),
        "%"
      )
    ),
    vjust = -0.3
  ) +
  scale_y_continuous(
    limits = c(
      0,
      1
    ),
    labels = scales::percent
  ) +
  labs(
    title = "Accuracy Comparison",
    x = "Machine Learning Algorithm",
    y = "Accuracy"
  ) +
  theme_minimal()


print(accuracy_plot)


ggsave(
  "08_accuracy_comparison.png",
  accuracy_plot,
  width = 8,
  height = 5
)


# ============================================================
# 23. PERFORMANCE METRICS COMPARISON
# ============================================================

comparison_long <- comparison %>%
  pivot_longer(
    cols = c(
      Accuracy,
      Precision,
      Recall,
      F1_Score
    ),
    names_to = "Metric",
    values_to = "Score"
  )


metric_plot <- ggplot(
  comparison_long,
  aes(
    x = Metric,
    y = Score,
    fill = Algorithm
  )
) +
  geom_col(
    position = "dodge"
  ) +
  scale_y_continuous(
    limits = c(
      0,
      1
    ),
    labels = scales::percent
  ) +
  labs(
    title = "Performance Comparison of ML Algorithms",
    x = "Evaluation Metric",
    y = "Score"
  ) +
  theme_minimal()


print(metric_plot)


ggsave(
  "09_model_performance_comparison.png",
  metric_plot,
  width = 9,
  height = 6
)


# ============================================================
# 24. DECISION TREE CONFUSION MATRIX
# ============================================================

tree_cm_table <- as.data.frame(
  table(
    Actual = test_data$Performance,
    Predicted = tree_prediction
  )
)


tree_cm_plot <- ggplot(
  tree_cm_table,
  aes(
    x = Predicted,
    y = Actual,
    fill = Freq
  )
) +
  geom_tile() +
  geom_text(
    aes(
      label = Freq
    ),
    size = 6
  ) +
  labs(
    title = "Decision Tree Confusion Matrix",
    x = "Predicted Class",
    y = "Actual Class"
  ) +
  theme_minimal()


print(tree_cm_plot)


ggsave(
  "10_decision_tree_confusion_matrix.png",
  tree_cm_plot,
  width = 8,
  height = 5
)


# ============================================================
# 25. RANDOM FOREST CONFUSION MATRIX
# ============================================================

rf_cm_table <- as.data.frame(
  table(
    Actual = test_data$Performance,
    Predicted = rf_prediction
  )
)


rf_cm_plot <- ggplot(
  rf_cm_table,
  aes(
    x = Predicted,
    y = Actual,
    fill = Freq
  )
) +
  geom_tile() +
  geom_text(
    aes(
      label = Freq
    ),
    size = 6
  ) +
  labs(
    title = "Random Forest Confusion Matrix",
    x = "Predicted Class",
    y = "Actual Class"
  ) +
  theme_minimal()


print(rf_cm_plot)


ggsave(
  "11_random_forest_confusion_matrix.png",
  rf_cm_plot,
  width = 8,
  height = 5
)


# ============================================================
# 26. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance_data <- importance(
  random_forest
)


importance_df <- data.frame(
  Feature = rownames(
    importance_data
  ),
  Importance = importance_data[
    ,
    "MeanDecreaseGini"
  ]
)


importance_df <- importance_df %>%
  arrange(
    desc(Importance)
  ) %>%
  head(10)


print(importance_df)


importance_plot <- ggplot(
  importance_df,
  aes(
    x = reorder(
      Feature,
      Importance
    ),
    y = Importance
  )
) +
  geom_col(
    fill = "darkorange"
  ) +
  coord_flip() +
  labs(
    title = "Top 10 Important Features - Random Forest",
    x = "Feature",
    y = "Importance"
  ) +
  theme_minimal()


print(importance_plot)


ggsave(
  "12_random_forest_feature_importance.png",
  importance_plot,
  width = 9,
  height = 6
)


# ============================================================
# 27. PREDICTION FOR TEST DATA
# ============================================================

prediction_results <- data.frame(
  Actual = test_data$Performance,
  Decision_Tree = tree_prediction,
  Random_Forest = rf_prediction
)


head(
  prediction_results,
  20
)


# ============================================================
# 28. PREDICT A NEW PLAYER
# ============================================================

# Enter the values for a new player below.
# The values should follow the same feature order
# used during model training.


new_player <- data.frame(
  
  Age = 25,
  
  Potential = 85,
  
  International_Reputation = 3,
  
  Weak_Foot = 4,
  
  Skill_Moves = 4,
  
  Crossing = 80,
  
  Finishing = 85,
  
  HeadingAccuracy = 70,
  
  ShortPassing = 85,
  
  Volleys = 80,
  
  Dribbling = 88,
  
  Curve = 75,
  
  FKAccuracy = 70,
  
  LongPassing = 80,
  
  BallControl = 88,
  
  Acceleration = 85,
  
  SprintSpeed = 84,
  
  Agility = 86,
  
  Reactions = 90,
  
  Balance = 80,
  
  ShotPower = 85,
  
  Jumping = 70,
  
  Stamina = 85,
  
  Strength = 70,
  
  LongShots = 82,
  
  Aggression = 50,
  
  Interceptions = 40,
  
  Positioning = 88,
  
  Vision = 85,
  
  Penalties = 70,
  
  Composure = 85,
  
  Marking = 35,
  
  StandingTackle = 30,
  
  SlidingTackle = 25
)


# ------------------------------------------------------------
# Decision Tree prediction for new player
# ------------------------------------------------------------

new_player_tree_prediction <- predict(
  decision_tree,
  new_player,
  type = "class"
)


# ------------------------------------------------------------
# Random Forest prediction for new player
# ------------------------------------------------------------

new_player_rf_prediction <- predict(
  random_forest,
  new_player
)


cat(
  "\n====================================\n"
)

cat(
  "NEW PLAYER PREDICTION\n"
)

cat(
  "====================================\n"
)


cat(
  "Decision Tree Prediction:",
  as.character(
    new_player_tree_prediction
  ),
  "\n"
)


cat(
  "Random Forest Prediction:",
  as.character(
    new_player_rf_prediction
  ),
  "\n"
)


# ============================================================
# 29. PREDICTION PROBABILITY
# ============================================================

tree_probability <- predict(
  decision_tree,
  new_player,
  type = "prob"
)


rf_probability <- predict(
  random_forest,
  new_player,
  type = "prob"
)


cat(
  "\nDecision Tree Probability:\n"
)

print(
  tree_probability
)


cat(
  "\nRandom Forest Probability:\n"
)

print(
  rf_probability
)


# ============================================================
# 30. FINAL MODEL COMPARISON
# ============================================================

print(
  comparison
)


cat(
  "\n====================================\n"
)

cat(
  "FINAL MODEL COMPARISON\n"
)

cat(
  "====================================\n"
  
  
)

cat(
  "Decision Tree Accuracy:",
  round(
    tree_accuracy * 100,
    2
  ),
  "%\n"
)


cat(
  "Random Forest Accuracy:",
  round(
    rf_accuracy * 100,
    2
  ),
  "%\n"
)


cat(
  "Decision Tree F1 Score:",
  round(
    tree_f1 * 100,
    2
  ),
  "%\n"
)


cat(
  "Random Forest F1 Score:",
  round(
    rf_f1 * 100,
    2
  ),
  "%\n"
)


cat(
  "\nProject completed successfully.\n"
)


cat(
  "\nGraphs saved in:\n"
)


cat(
  getwd(),
  "\n"
)