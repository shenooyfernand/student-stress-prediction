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

# Assumes you already built xtrain, xtest, ytrain, ytest,
# and have compute_f1() + aggregate_importance() from your previous script.

library(caret)
library(kernlab)
library(ggplot2)

## ---------- 1) SVM — Linear ----------
ctrl_svm_lin <- trainControl(method = "repeatedcv",
                             number = 5, repeats = 2,
                             classProbs = TRUE,
                             verboseIter = TRUE,
                             selectionFunction = "oneSE")

grid_svm_lin <- expand.grid(C = 2^seq(-3, 3, by = 1))

svm_lin_tuned <- train(x = xtrain, y = ytrain,
                       method = "svmLinear",
                       trControl = ctrl_svm_lin,
                       tuneGrid = grid_svm_lin,
                       metric = "Accuracy")

svm_lin_final <- train(x = xtrain, y = ytrain,
                       method = "svmLinear",
                       trControl = trainControl(method="none", classProbs=TRUE),
                       tuneGrid = svm_lin_tuned$bestTune)

pred_lin_test  <- predict(svm_lin_final, xtest)
pred_lin_train <- predict(svm_lin_final, xtrain)

cm_lin_test  <- confusionMatrix(pred_lin_test,  ytest)
cm_lin_train <- confusionMatrix(pred_lin_train, ytrain)
print(cm_lin_test); print(cm_lin_train)
print(compute_f1(cm_lin_test)); print(compute_f1(cm_lin_train))

vi_lin <- varImp(svm_lin_final, scale = FALSE)$importance
vi_lin$Feature <- rownames(vi_lin); colnames(vi_lin)[1] <- "Overall"
agg_lin <- aggregate_importance(vi_lin); print(agg_lin)

ggplot(agg_lin, aes(x=reorder(Base, AvgScaled), y=AvgScaled)) +
  geom_col() + coord_flip() +
  labs(title="SVM Linear: Aggregated Feature Importance", x="Feature group", y="Aggregated importance") +
  theme_minimal(base_size = 14)


## ---------- 2) SVM — Radial (RBF) ----------
ctrl_svm_rbf <- trainControl(method = "repeatedcv",
                             number = 5, repeats = 2,
                             classProbs = TRUE,
                             verboseIter = TRUE,
                             selectionFunction = "oneSE")

sig0 <- kernlab::sigest(~., data = as.data.frame(xtrain))[[1]]

grid_svm_rbf <- expand.grid(
  sigma = sig0 * c(0.5, 1, 2),
  C = 2^c(-1, 0, 1, 2)
)

svm_rbf_tuned <- train(x = xtrain, y = ytrain,
                       method = "svmRadial",
                       trControl = ctrl_svm_rbf,
                       tuneGrid = grid_svm_rbf,
                       metric = "Accuracy")

svm_rbf_final <- train(x = xtrain, y = ytrain,
                       method = "svmRadial",
                       trControl = trainControl(method="none", classProbs=TRUE),
                       tuneGrid = svm_rbf_tuned$bestTune)

pred_rbf_test  <- predict(svm_rbf_final, xtest)
pred_rbf_train <- predict(svm_rbf_final, xtrain)

cm_rbf_test  <- confusionMatrix(pred_rbf_test,  ytest)
cm_rbf_train <- confusionMatrix(pred_rbf_train, ytrain)
print(cm_rbf_test); print(cm_rbf_train)
print(compute_f1(cm_rbf_test)); print(compute_f1(cm_rbf_train))

vi_rbf <- varImp(svm_rbf_final, scale = FALSE)$importance
vi_rbf$Feature <- rownames(vi_rbf); colnames(vi_rbf)[1] <- "Overall"
agg_rbf <- aggregate_importance(vi_rbf); print(agg_rbf)

ggplot(agg_rbf, aes(x=reorder(Base, AvgScaled), y=AvgScaled)) +
  geom_col() + coord_flip() +
  labs(title="SVM RBF: Aggregated Feature Importance", x="Feature group", y="Aggregated importance") +
  theme_minimal(base_size = 14)


## ---------- 3) SVM — Polynomial ----------
ctrl_svm_poly <- trainControl(method = "repeatedcv",
                              number = 5, repeats = 2,
                              classProbs = TRUE,
                              verboseIter = TRUE,
                              selectionFunction = "oneSE")

grid_svm_poly <- expand.grid(
  degree = c(2, 3, 4),
  scale  = 10^seq(-3, 0, length.out = 4),  # gamma in polydot
  C      = 2^c(-1, 0, 1, 2)
)

svm_poly_tuned <- train(x = xtrain, y = ytrain,
                        method = "svmPoly",
                        trControl = ctrl_svm_poly,
                        tuneGrid = grid_svm_poly,
                        metric = "Accuracy")

svm_poly_final <- train(x = xtrain, y = ytrain,
                        method = "svmPoly",
                        trControl = trainControl(method="none", classProbs=TRUE),
                        tuneGrid = svm_poly_tuned$bestTune)

pred_poly_test  <- predict(svm_poly_final, xtest)
pred_poly_train <- predict(svm_poly_final, xtrain)

cm_poly_test  <- confusionMatrix(pred_poly_test,  ytest)
cm_poly_train <- confusionMatrix(pred_poly_train, ytrain)
print(cm_poly_test); print(cm_poly_train)
print(compute_f1(cm_poly_test)); print(compute_f1(cm_poly_train))

vi_poly <- varImp(svm_poly_final, scale = FALSE)$importance
vi_poly$Feature <- rownames(vi_poly); colnames(vi_poly)[1] <- "Overall"
agg_poly <- aggregate_importance(vi_poly); print(agg_poly)

ggplot(agg_poly, aes(x=reorder(Base, AvgScaled), y=AvgScaled)) +
  geom_col() + coord_flip() +
  labs(title="SVM Polynomial: Aggregated Feature Importance", x="Feature group", y="Aggregated importance") +
  theme_minimal(base_size = 14)


## ---------- 4) SVM — Sigmoid (tanh kernel, via kernlab::ksvm + manual grid) ----------
# caret has no built-in "svmSigmoid" method; we tune ksvm directly and compute permutation importance.

perm_importance <- function(model, x_val, y_val, predict_fun, repeats = 1L) {
  base_pred <- predict_fun(model, x_val)
  base_acc  <- mean(base_pred == y_val)
  out <- data.frame(Feature = colnames(x_val), Drop = NA_real_)
  set.seed(123)
  for (j in seq_len(ncol(x_val))) {
    drops <- numeric(repeats)
    for (r in seq_len(repeats)) {
      x_perm <- x_val
      x_perm[, j] <- sample(x_perm[, j])  # permute a single feature
      pred_p <- predict_fun(model, x_perm)
      drops[r] <- base_acc - mean(pred_p == y_val)
    }
    out$Drop[j] <- mean(drops)
  }
  out[order(-out$Drop), , drop = FALSE]
}

# Grid over C, scale, and offset for tanhdot
C_grid      <- 2^c(-1, 0, 1, 2)
scale_grid  <- 10^seq(-3, -1, length.out = 3)
offset_grid <- c(0, 1)

folds <- createFolds(ytrain, k = 5, returnTrain = FALSE)
best_acc <- -Inf
best_par <- NULL

for (Cval in C_grid) {
  for (sc in scale_grid) {
    for (off in offset_grid) {
      accs <- c()
      for (fi in seq_along(folds)) {
        val_idx <- folds[[fi]]
        tr_idx  <- setdiff(seq_len(nrow(xtrain)), val_idx)
        model_cv <- ksvm(x = xtrain[tr_idx, , drop=FALSE],
                         y = ytrain[tr_idx],
                         kernel = "tanhdot",
                         kpar = list(scale = sc, offset = off),
                         C = Cval,
                         type = "C-svc",
                         prob.model = FALSE)
        pred_cv <- predict(model_cv, xtrain[val_idx, , drop=FALSE])
        accs[fi] <- mean(pred_cv == ytrain[val_idx])
      }
      macc <- mean(accs)
      if (macc > best_acc) {
        best_acc <- macc
        best_par <- list(C = Cval, scale = sc, offset = off)
      }
    }
  }
}

svm_sig_final <- ksvm(x = xtrain, y = ytrain,
                      kernel = "tanhdot",
                      kpar = list(scale = best_par$scale, offset = best_par$offset),
                      C = best_par$C,
                      type = "C-svc",
                      prob.model = FALSE)

pred_sig_test  <- predict(svm_sig_final, xtest)
pred_sig_train <- predict(svm_sig_final, xtrain)

cm_sig_test  <- confusionMatrix(pred_sig_test,  ytest)
cm_sig_train <- confusionMatrix(pred_sig_train, ytrain)
print(cm_sig_test); print(cm_sig_train)
print(compute_f1(cm_sig_test)); print(compute_f1(cm_sig_train))

# Permutation importance on test set, then aggregate by base variable (pre-dummy)
vi_sig <- perm_importance(svm_sig_final, xtest, ytest,
                          predict_fun = function(m, X) predict(m, X), repeats = 1)
colnames(vi_sig)[2] <- "Overall"
vi_sig$Base <- sub("\\..*$", "", vi_sig$Feature)
agg_sig <- aggregate_importance(vi_sig[, c("Feature","Overall","Base")]); print(agg_sig)

ggplot(agg_sig, aes(x=reorder(Base, AvgScaled), y=AvgScaled)) +
  geom_col() + coord_flip() +
  labs(title="SVM Sigmoid (tanhdot): Aggregated Permutation Importance",
       x="Feature group", y="Aggregated importance") +
  theme_minimal(base_size = 14)
