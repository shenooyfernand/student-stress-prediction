# Clear environment
rm(list = ls())

# Reproducibility
set.seed(123)

# Load dataset
data <- read.csv("C:/Users/sheno/OneDrive/Desktop/St4052/Final/Dataset/StressLevelDataset.csv")

# Basic summary
summary(data)
nrow(data)
ncol(data)

# Remove duplicate rows
data1 <- data[!duplicated(data), ]

# Remove rows with missing values
data2 <- na.omit(data1)

# Verify cleaning
cat("Rows before:", nrow(data), "\n")
cat("Rows after removing duplicates:", nrow(data1), "\n")
cat("Rows after removing NA:", nrow(data2), "\n")

n=nrow(data2)

for (i in 1:n){
  if (data2$stress_level[i]==0){data2$stress_level[i]="No_Stress"}
  if (data2$stress_level[i]==1){data2$stress_level[i]="Eustress"}
  if (data2$stress_level[i]==2){data2$stress_level[i]="Distress"}
}
data2$stress_level=as.factor(data2$stress_level)

train_index=sample(n,n*0.8)
train=data2[train_index,]
test =data2[-train_index,]

#histogram
library(ggplot2)

# Identify numeric columns
num_cols <- sapply(train, is.numeric)
numeric_vars <- names(train)[num_cols]

# Create histograms for each numeric variable
for (col in numeric_vars) {
  p <- ggplot(train, aes(x = .data[[col]])) +
    geom_histogram(bins = 30, fill = "#69b3a2", color = "black") +
    labs(title = paste("Distribution of", col),
         x = col, y = "Frequency") +
    theme_minimal()
  print(p)
}

train$stress_level=as.factor(train$stress_level)
library(dplyr)

# Count proportions of each stress type
stress_counts <- train %>%
  group_by(stress_level) %>%
  summarise(count = n()) %>%
  mutate(percent = round(100 * count / sum(count), 1),
         label = paste0(stress_level, "\n", percent, "%"))

# Pie chart
ggplot(stress_counts, aes(x = "", y = percent, fill = stress_level)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar(theta = "y") +
  labs(title = "Distribution of Stress Type (Train Set)", x = NULL, y = NULL) +
  theme_void() +
  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5),
            color = "white", size = 4) +
  scale_fill_brewer(palette = "Set2")


# Identify numeric predictors (exclude target column)
num_cols <- names(train)[sapply(train, is.numeric)]
num_cols <- setdiff(num_cols, "stress_type")  # skip if stress_type is numeric-coded

# Loop through each numeric variable
for (col in num_cols) {
  p <- ggplot(train, aes(x = stress_level, y = .data[[col]], fill = stress_level)) +
    geom_boxplot(outlier.color = "red", outlier.size = 1.5, alpha = 0.8) +
    labs(title = paste("Boxplot of", col, "by Stress level"),
         x = "Stress level", y = col) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold", hjust = 0.5))
  print(p)
}
