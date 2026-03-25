rm(list=ls())
library(caret)
library(xgboost)
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

train_control <- trainControl(method = "repeatedcv",
                              number = 5,
                              repeats = 2,
                              classProbs = TRUE,
                              verboseIter = TRUE,
                              selectionFunction = "oneSE")

grid_tune <- expand.grid(
  nrounds = 100,
  max_depth = 3,
  eta = 0.01,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
) 

xgb_tune <- train(x = xtrain,
                  y = ytrain,
                  trControl = train_control,
                  tuneGrid = grid_tune,
                  method = "xgbTree",
                  metric = "Accuracy",
                  verbose = TRUE)

final_train_control <- trainControl(method = "none", classProbs = TRUE, verboseIter = TRUE)

xgb_best_model <- train(x = xtrain,
                        y = ytrain,
                        trControl = final_train_control,
                        tuneGrid = xgb_tune$bestTune,
                        method = "xgbTree",
                        verbose = TRUE)

pred_test  <- predict(xgb_best_model, xtest)
pred_train <- predict(xgb_best_model, xtrain)

cm_test  <- confusionMatrix(pred_test, ytest)
cm_train <- confusionMatrix(pred_train, ytrain)
print(cm_test)
print(cm_train)

compute_f1 <- function(cm) { p <- cm$byClass[, "Pos Pred Value"]; r <- cm$byClass[, "Sensitivity"]; 2*(p*r)/(p+r) }
print(compute_f1(cm_test))
print(compute_f1(cm_train))

imp <- xgb.importance(model = xgb_best_model$finalModel)
map_base <- function(f, bases) {
  lens <- ifelse(startsWith(f, bases), nchar(bases), -1)
  if (max(lens) < 0) f else bases[which.max(lens)]
}
all_bases <- c(num_cols, cat_cols)
imp$Base <- vapply(imp$Feature, map_base, character(1), bases = all_bases)

agg_gain <- aggregate(Gain ~ Base, data = imp, sum)
agg_cover <- aggregate(Cover ~ Base, data = imp, sum)
agg_freq <- aggregate(Frequency ~ Base, data = imp, sum)
agg <- Reduce(function(a,b) merge(a,b,by="Base",all=TRUE), list(agg_gain, agg_cover, agg_freq))
scale01 <- function(x) (x - min(x, na.rm=TRUE)) / (max(x, na.rm=TRUE) - min(x, na.rm=TRUE) + 1e-12)
agg$Gain_scaled <- scale01(agg$Gain)
agg$Cover_scaled <- scale01(agg$Cover)
agg$Freq_scaled <- scale01(agg$Frequency)
agg$AvgScaled <- rowMeans(cbind(agg$Gain_scaled, agg$Cover_scaled, agg$Freq_scaled), na.rm = TRUE)
agg <- agg[order(-agg$AvgScaled), ]

print(agg)

ggplot(agg, aes(x = reorder(Base, AvgScaled), y = AvgScaled)) +
  geom_col() +
  coord_flip() +
  labs(title = "Aggregated Feature Importance (Avg scaled Gain/Cover/Frequency)",
       x = "Feature (grouped by original variable)", y = "Aggregated importance") +
  theme_minimal(base_size = 14)

library(pdp)
library(ggplot2)

pdp_df <- train[, c(cat_cols, num_cols)]

pred_fun <- function(object, newdata){
  X <- as.matrix(predict(dmy, newdata = newdata[, c(cat_cols, num_cols), drop = FALSE]))
  probs <- predict(object, X, type = "prob")
  as.numeric(probs[["Distress"]])
}

make_pdp <- function(var, grid.res = 20, with_ice = TRUE){
  pd <- partial(
    object = xgb_best_model,
    pred.var = var,
    pred.fun = pred_fun,
    train = pdp_df,            # keep this so pdp knows your data
    grid.resolution = if (var %in% num_cols) grid.res else NULL,
    ice = with_ice,            # ICE on/off
    center = with_ice          # center ICE if on
  )
  autoplot(pd,
           rug = FALSE,        # <- avoids the error
           alpha = if (with_ice) 0.15 else 1,
           plot.pdp = TRUE) +
    labs(title = paste0("P(Distress) vs ", var),
         x = var, y = "Predicted probability") +
    theme_minimal(base_size = 13)
}

all_vars <- c(num_cols, cat_cols)
plots <- setNames(lapply(all_vars, make_pdp), all_vars)

dir.create("pdp_plots", showWarnings = FALSE)
for (v in names(plots)) {
  ggsave(file.path("pdp_plots", paste0("pdp_", v, ".png")),
         plots[[v]], width = 6.5, height = 4.2, dpi = 300)
}



