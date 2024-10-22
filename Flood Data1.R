

# Load necessary libraries

install.packages("readr")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("caret")
install.packages("randomForest")
install.packages("gbm")
install.packages("xgboost")
install.packages("lightgbm")
install.packages("e1071")
install.packages('gridExtra')

library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(gbm)
library(xgboost)
library(lightgbm)
library(e1071)
library(gridExtra)
# Load the dataset

flood <- read_csv("C:/Users/Lib 003/Desktop/Kaggle Website Research/Flood Data/flood.csv")


# Inspect the dataset

str(flood)
head(flood)
summary(flood)
is.na(flood)

##################

is.factor(flood$FloodProbability)

##### floodprobability column is not a factor so, we have to convert it to a factor.


########## flood probability distribution
########I have not gotten the result to this code



# Load necessary libraries
library(ggplot2)
library(gridExtra)
library(dplyr)




# Remove 'FloodProbability' column from the columns list
columns <- setdiff(names(flood), "FloodProbability")

# Create a list to store the plots
plots <- list()

# Loop through the columns to create individual histograms
for (col in columns) {
  # Get column data
  col_data <- flood[[col]]
  
  # Create histogram
  p <- ggplot(flood, aes_string(x = col)) +
    geom_histogram(binwidth = 1, fill = "green", color = "black", alpha = 0.7) +
    labs(title = paste("Histogram of", col), x = "Value", y = "Frequency") +
    theme_minimal()
  
  # Add summary statistics text
  summary_stats <- summary(col_data)
  summary_text <- paste(names(summary_stats), sprintf("%.2f", summary_stats), sep = ": ", collapse = "\n")
  
  p <- p + annotate("text", x = Inf, y = Inf, label = summary_text, hjust = 1.1, vjust = 1.1, size = 3, 
                    color = "black", fontface = "italic", family = "Courier")
  
  # Store the plot in the list
  plots[[col]] <- p
}

# Split plots into two sets of 10
plots_set1 <- plots[1:10]
plots_set2 <- plots[11:20]

# Calculate the number of rows and columns for the grid
n_cols <- 2
n_rows <- ceiling(length(plots_set1) / n_cols)

# Arrange and display the first set of plots in a grid layout
do.call(grid.arrange, c(plots_set1, ncol = n_cols))

# Arrange and display the second set of plots in a grid layout
do.call(grid.arrange, c(plots_set2, ncol = n_cols))





###########################################


# Calculate the correlation matrix

# Install and load ggcorrplot
install.packages("ggcorrplot")
library(ggcorrplot)




# Ensure numeric columns are selected for correlation matrix

numeric_columns <- flood %>% select_if(is.numeric)

# Calculate the correlation matrix

correlation_matrix <- cor(numeric_columns, use = "complete.obs")

# Plot the correlation matrix

ggcorrplot(correlation_matrix, method = "circle", type = "lower", 
           lab = TRUE, lab_size = 3, colors = c("red", "white", "blue"), 
           title = "Correlation Matrix", ggtheme = theme_minimal())


########################################### I want to scale the data


library(caret)    # For feature scaling



# Separate features and target variable

x <- flood %>% select(-FloodProbability)
y <- flood$FloodProbability

# Initialize the MinMaxScaler equivalent in R (range from 0 to 1)
###### I do not wish to scale the data before splitting because I am not getting the required result.

#scaler <- preProcess(x, method = c("range"))

# Fit and transform the features
#x_scaled <- predict(scaler, x)

# Convert the scaled features to a data frame
#x_scaled_df <- as.data.frame(x_scaled)

# Print the first few rows of the scaled data frame
#head(x_scaled_df)
###########################################

########################## splitting the data but i have to combine it with the scaled one




library(caret)
library(randomForest)
library(ggplot2)
library(gridExtra)




# Set the seed for reproducibility
set.seed(42)

# Split the data into training and testing sets (80% training, 20% testing)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)

# Create training and testing sets
x_train <- x[trainIndex, ]
x_test <- x[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]


# Print the dimensions of the training and testing sets
cat("Training set dimensions: ", dim(x_train), "\n")
cat("Testing set dimensions: ", dim(x_test), "\n")



################################








# Load required libraries
library(randomForest)
library(gbm)
library(xgboost)
library(lightgbm)
library(caret)
library(e1071)  # For SVM
library(ggplot2)




# Define the models
models <- list(
  Random_Forest = randomForest(x_train, y_train, ntree=100),
  Gradient_Boosting = gbm(y_train ~ ., data = data.frame(x_train, y_train), distribution = "gaussian", n.trees = 100),
  Linear_Regression = lm(y_train ~ ., data = data.frame(x_train, y_train))
)

# Names of the models
names <- c("Random Forest", "Gradient Boosting", "Linear Regression")

# Initialize vectors to store results
r2s <- numeric(length(models))
mses <- numeric(length(models))

# Train models and calculate metrics
for (i in seq_along(models)) {
  model <- models[[i]]
  
  # Print the name of the model being trained
  cat("Training model:", names[i], "\n")
  
  pred <- predict(model, x_test)
  
  r2s[i] <- caret::R2(pred, y_test) * 100
  mses[i] <- mean((pred - y_test)^2)
}

# Create a data frame for the results
results <- data.frame(Model = names, R2 = r2s, MSE = mses)

# Plot the results
ggplot(results, aes(x = reorder(Model, -R2), y = R2)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  geom_text(aes(label = round(R2, 2)), vjust = -0.3) +
  labs(title = "R2 Scores of Models", x = "", y = "R2 (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(results, aes(x = reorder(Model, MSE), y = MSE)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  geom_text(aes(label = round(MSE, 2)), vjust = -0.3) +
  labs(title = "MSE Scores of Models", x = "", y = "Mean Squared Error") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

















###################################################Confusion Matrix and Statistics


install.packages(c("randomForest", "gbm", "caret"))
install.packages("Metrics")

library(Metrics)
library(randomForest)
library(gbm)
library(caret)

#install.packages("caTools")
#library(caTools)



flood <- read_csv("C:/Users/Lib 003/Desktop/Kaggle Website Research/Flood Data/flood.csv")




# Split the data into training and testing sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(flood$FloodProbability, p = 0.8, list = FALSE)
train_data <- flood[train_index, ]
test_data <- flood[-train_index, ]

##############Linear regression classification


# Train a linear regression model
lm_model <- lm(FloodProbability ~ ., data = train_data)

# Make predictions
lm_pred <- predict(lm_model, newdata = test_data)

# Calculate metrics
library(Metrics)
mae <- mae(test_data$FloodProbability, lm_pred)
mse <- mse(test_data$FloodProbability, lm_pred)
rmse <- sqrt(mse)
r_squared <- caret::R2(test_data$FloodProbability, lm_pred)

# Print metrics
cat("Linear Regression Metrics:\n")
cat("MAE:", mae, "\n")
cat("MSE:", mse, "\n")
cat("RMSE:", rmse, "\n")
cat("R-squared:", r_squared, "\n")


################################Random forest classification


# Install necessary packages
install.packages("randomForest")
install.packages("gbm")
install.packages("caret")
install.packages("e1071")




# Install necessary packages
install.packages("randomForest")
install.packages("gbm")
install.packages("caret")
install.packages("e1071")

# Load the libraries
library(randomForest)
library(gbm)
library(caret)
library(e1071)

# Load your flood dataset
# Assuming your dataset is already loaded as flood_data
# Ensure the response variable is a factor for classification
flood$FloodProbability <- as.factor(flood$FloodProbability)

# Split the data into training and testing sets
set.seed(42)
train_index <- createDataPartition(flood$FloodProbability, p = 0.8, list = FALSE)
train_data <- flood[train_index, ]
test_data <- flood[-train_index, ]

# Train the Random Forest model
rf_model <- randomForest(FloodProbability ~ ., data = train_data)

# Train the Gradient Boosting model
gb_model <- gbm(FloodProbability ~ ., data = train_data, distribution = "multinomial", n.trees = 100)

# Make predictions with the Random Forest model
rf_predictions <- predict(rf_model, test_data)

# Make predictions with the Gradient Boosting model
gb_predictions <- predict(gb_model, newdata = test_data, n.trees = 100, type = "response")
gb_predictions <- apply(gb_predictions, 1, which.max)
gb_predictions <- factor(gb_predictions, levels = 1:length(unique(flood$FloodProbability)), labels = levels(flood$FloodProbability))

# Evaluate the Random Forest model
print("Random Forest Classifier:")
rf_conf_matrix <- confusionMatrix(rf_predictions, test_data$FloodProbability)
print(rf_conf_matrix)

# Evaluate the Gradient Boosting model
print("Gradient Boosting Classifier:")
gb_conf_matrix <- confusionMatrix(gb_predictions, test_data$FloodProbability)
print(gb_conf_matrix)







######################################################################


# Install necessary packages
install.packages("randomForest")
install.packages("gbm")
install.packages("caret")
install.packages("e1071")
install.packages("plotly")

# Load the libraries
library(randomForest)
library(gbm)
library(caret)
library(plotly)

# Assuming 'Flood_Prediction' data frame is already defined and loaded
# Define independent and dependent variables
X <- flood[, c('MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
                          'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
                          'Siltation', 'AgriculturalPractices', 'Encroachments',
                          'IneffectiveDisasterPreparedness', 'DrainageSystems',
                          'CoastalVulnerability', 'Landslides', 'Watersheds',
                          'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
                          'InadequatePlanning', 'PoliticalFactors')]
Y <- flood$FloodProbability

# Split the data
set.seed(42)
train_index <- createDataPartition(Y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
Y_train <- Y[train_index]
X_test <- X[-train_index, ]
Y_test <- Y[-train_index]

# Random Forest model
rf_model <- randomForest(X_train, Y_train, importance=TRUE, ntree=500)
rf_predictions <- predict(rf_model, X_test)

# Evaluation metrics for Random Forest
rf_mse <- mean((rf_predictions - Y_test)^2)
rf_mae <- mean(abs(rf_predictions - Y_test))
rf_r2 <- 1 - sum((rf_predictions - Y_test)^2) / sum((Y_test - mean(Y_test))^2)

cat("Random Forest Test MSE: ", rf_mse, "\n")
cat("Random Forest Test MAE: ", rf_mae, "\n")
cat("Random Forest Test R²: ", rf_r2, "\n")

# Feature importances for Random Forest
rf_importances <- data.frame(Feature = row.names(rf_model$importance), Importance = rf_model$importance[, 'IncNodePurity'])
plot_ly(rf_importances, x = ~Feature, y = ~Importance, type = 'bar', name = 'Random Forest') %>%
  layout(title = 'Random Forest Feature Importances', xaxis = list(title = 'Features'), yaxis = list(title = 'Importance'))

# Gradient Boosting model
gb_model <- gbm(FloodProbability ~ ., data = Flood_Prediction[train_index,], distribution = "gaussian", n.trees = 5000, interaction.depth = 4, shrinkage = 0.01, cv.folds = 10)
best_iter <- gbm.perf(gb_model, method = "cv")
gb_predictions <- predict(gb_model, X_test, n.trees = best_iter)

# Evaluation metrics for Gradient Boosting
gb_mse <- mean((gb_predictions - Y_test)^2)
gb_mae <- mean(abs(gb_predictions - Y_test))
gb_r2 <- 1 - sum((gb_predictions - Y_test)^2) / sum((Y_test - mean(Y_test))^2)

cat("Gradient Boosting Test MSE: ", gb_mse, "\n")
cat("Gradient Boosting Test MAE: ", gb_mae, "\n")
cat("Gradient Boosting Test R²: ", gb_r2, "\n")

# Feature importances for Gradient Boosting
summary(gb_model, n.trees = best_iter)

# Interactive plot for Actual vs. Predicted Values for Random Forest
fig <- plot_ly(x = ~Y_test, y = ~rf_predictions, type = 'scatter', mode = 'markers',
               marker = list(color = 'rgba(152, 0, 0, .8)')) %>%
  layout(title = 'Actual vs. Predicted Flood Probability (Random Forest)',
         xaxis = list(title = 'Actual Flood Probability (%)'),
         yaxis = list(title = 'Predicted Flood Probability (%)')) %>%
  add_lines(x = ~Y_test, y = ~Y_test, line = list(dash = 'dash', color = 'red'))

fig

# Interactive plot for Actual vs. Predicted Values for Gradient Boosting
fig <- plot_ly(x = ~Y_test, y = ~gb_predictions, type = 'scatter', mode = 'markers',
               marker = list(color = 'rgba(0, 152, 0, .8)')) %>%
  layout(title = 'Actual vs. Predicted Flood Probability (Gradient Boosting)',
         xaxis = list(title = 'Actual Flood Probability (%)'),
         yaxis = list(title = 'Predicted Flood Probability (%)')) %>%
  add_lines(x = ~Y_test, y = ~Y_test, line = list(dash = 'dash', color = 'red'))

fig

# Comparison DataFrame for Random Forest
comparison_rf <- data.frame('Actual Flood Probability (%)' = Y_test, 'Predicted Flood Probability (%)' = rf_predictions)
print(head(comparison_rf))

# Comparison DataFrame for Gradient Boosting
comparison_gb <- data.frame('Actual Flood Probability (%)' = Y_test, 'Predicted Flood Probability (%)' = gb_predictions)
print(head(comparison_gb))

# Save the Random Forest model
saveRDS(rf_model, 'random_forest_flood_model.rds')

# Save the Gradient Boosting model
saveRDS(gb_model, 'gradient_boosting_flood_model.rds')



