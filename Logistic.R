rm(list=ls())
library(caret)
library(ggplot2)
set.seed(123)

train <- read.csv("C:/Users/sheno/OneDrive/Desktop/St4052/Final/Dataset/New_cluster_train.csv")
test  <- read.csv("C:/Users/sheno/OneDrive/Desktop/St4052/Final/Dataset/New_cluster_test.csv")

label_map <- c("No_Stress"=0, "Eustress"=1, "Distress"=2)
train$stress_level_fac <- factor(as.character(train$stress_level), levels = names(label_map))
test$ytest_fac <- factor(as.character(test$ytest), levels = names(label_map))

num_keep <- c("anxiety_levels","depression","self_esteem")
all_cols <- setdiff(names(train), c("stress_level","stress_level_fac"))
preds <- intersect(all_cols, names(test))
num_cols <- intersect(num_keep, preds)
cat_cols <- setdiff(preds, num_cols)

for (c in cat_cols) { train[[c]] <- as.factor(train[[c]]); test[[c]] <- as.factor(test[[c]]) }
for (c in cat_cols) {
  tr_lvls <- levels(train[[c]])
  if (!("Other" %in% tr_lvls)) tr_lvls <- c(tr_lvls, "Other")
  train[[c]] <- factor(train[[c]], levels = tr_lvls)
  tv <- as.character(test[[c]]); tv[!(tv %in% tr_lvls)] <- "Other"
  test[[c]] <- factor(tv, levels = tr_lvls)
}

dmy <- dummyVars(~ ., data = train[, c(cat_cols, num_cols)], fullRank = TRUE)
xtrain <- as.matrix(predict(dmy, newdata = train[, c(cat_cols, num_cols)]))
xtest  <- as.matrix(predict(dmy, newdata = test[, c(cat_cols, num_cols)]))
ytrain <- factor(train$stress_level_fac, levels = names(label_map))
ytest  <- factor(test$ytest_fac, levels = names(label_map))

compute_f1 <- function(cm) { p <- cm$byClass[, "Pos Pred Value"]; r <- cm$byClass[, "Sensitivity"]; 2*(p*r)/(p+r) }

aggregate_importance <- function(imp_df) {
  imp_df$Base <- sub("\\..*$", "", imp_df$Feature)
  agg_gain <- if ("Gain" %in% names(imp_df)) aggregate(Gain ~ Base, data = imp_df, sum) else NULL
  agg_cover <- if ("Cover" %in% names(imp_df)) aggregate(Cover ~ Base, data = imp_df, sum) else NULL
  agg_freq <- if ("Frequency" %in% names(imp_df)) aggregate(Frequency ~ Base, data = imp_df, sum) else NULL
  parts <- Filter(Negate(is.null), list(agg_gain, agg_cover, agg_freq))
  if (length(parts) == 0) parts <- list(aggregate(Overall ~ Base, data = imp_df, sum))
  agg <- Reduce(function(a,b) merge(a,b,by="Base",all=TRUE), parts)
  scale01 <- function(x) (x - min(x, na.rm=TRUE)) / (max(x, na.rm=TRUE) - min(x, na.rm=TRUE) + 1e-12)
  for (col in intersect(c("Gain","Cover","Frequency","Overall"), names(agg))) agg[[paste0(col,"_scaled")]] <- scale01(agg[[col]])
  scaled_cols <- grep("_scaled$", names(agg), value = TRUE)
  agg$AvgScaled <- rowMeans(agg[, scaled_cols, drop=FALSE], na.rm = TRUE)
  agg[order(-agg$AvgScaled), ]
}

library(glmnet)

ctrl_glm <- trainControl(method = "repeatedcv",
                         number = 5, repeats = 2,
                         classProbs = TRUE,
                         verboseIter = TRUE,
                         selectionFunction = "oneSE")

grid_glm <- expand.grid(
  alpha = c(0, 0.5, 1),
  lambda = 10^seq(-3, 0, length.out = 15)
)

glmnet_tuned <- train(x = xtrain, y = ytrain,
                      method = "glmnet",
                      trControl = ctrl_glm,
                      tuneGrid = grid_glm,
                      family = "multinomial",
                      standardize = TRUE)

glmnet_final <- train(x = xtrain, y = ytrain,
                      method = "glmnet",
                      trControl = trainControl(method="none", classProbs=TRUE),
                      tuneGrid = glmnet_tuned$bestTune,
                      family = "multinomial",
                      standardize = TRUE)

pred_glm_test  <- predict(glmnet_final, xtest)
pred_glm_train <- predict(glmnet_final, xtrain)

cm_glm_test  <- confusionMatrix(pred_glm_test,  ytest)
cm_glm_train <- confusionMatrix(pred_glm_train, ytrain)
print(cm_glm_test); print(cm_glm_train)
print(compute_f1(cm_glm_test)); print(compute_f1(cm_glm_train))

vi_glm <- varImp(glmnet_final, scale = FALSE)$importance
vi_glm$Feature <- rownames(vi_glm)
colnames(vi_glm)[1] <- "Overall"
agg_glm <- aggregate_importance(vi_glm)

print(agg_glm)
ggplot(agg_glm, aes(x=reorder(Base, AvgScaled), y=AvgScaled)) +
  geom_col() + coord_flip() +
  labs(title="Multinomial Logistic (glmnet): Aggregated Feature Importance", x="Feature", y="Aggregated importance") +
  theme_minimal(base_size = 14)
