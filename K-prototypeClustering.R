rm(list = ls())
set.seed(123)

library(clustMixType)
library(cluster)
library(dplyr)
library(ggplot2)
library(factoextra)

data = read.csv("C:/Users/sheno/OneDrive/Desktop/St4052/Final/Dataset/train.csv")
data = data[,-1]

label_candidates = intersect(names(data), c("stress_level","stress_type","Outcome"))
target = NULL
if (length(label_candidates) > 0) {
  target = data[, label_candidates, drop = FALSE]
  data   = data[, setdiff(names(data), label_candidates), drop = FALSE]
}
data1=data
numeric_cols = c("anxiety_level","depression","self_esteem")
stopifnot(all(numeric_cols %in% names(data)))
categorical_cols = setdiff(names(data), numeric_cols)

# FIX: Use as.data.frame() to convert list back to data frame
data[categorical_cols] = as.data.frame(lapply(data[categorical_cols], function(x) if (is.factor(x)) x else factor(x)))
data[numeric_cols] = as.data.frame(lapply(data[numeric_cols], function(x) as.numeric(as.character(x))))

data_scaled = data
data_scaled[numeric_cols] = as.data.frame(scale(data_scaled[numeric_cols]))

cat_idx = which(names(data_scaled) %in% categorical_cols)
lambda = tryCatch(clustMixType::lambdaest(data_scaled, catCols = cat_idx), error = function(e) NULL)
cat("Estimated lambda:", ifelse(is.null(lambda), "auto", round(lambda, 4)), "\n")

get_wss = function(dat, k.min=2, k.max=10, lambda=NULL) {
  ks = k.min:k.max
  wss = numeric(length(ks))
  for (i in seq_along(ks)) {
    set.seed(123)
    fit = kproto(dat, k = ks[i], lambda = lambda, nstart = 5)
    wss[i] = fit$tot.withinss
  }
  data.frame(k = ks, wss = wss)
}

elbow = get_wss(data_scaled, 2, 10, lambda)
ggplot(elbow, aes(k, wss)) + geom_line() + geom_point() +
  scale_x_continuous(breaks = elbow$k) +
  labs(title = "Elbow for K-Prototypes", x = "k", y = "Total withinss") +
  theme_minimal()

k_opt = 3
set.seed(123)
kmod = kproto(data_scaled, k = k_opt, lambda = lambda, nstart = 10)

data$kproto_cluster = factor(kmod$cluster)

cat("\nCluster sizes:\n"); print(table(data$kproto_cluster))
cat("\nCluster centers:\n"); print(kmod$centers)

gowerD = daisy(data[, c(numeric_cols, categorical_cols)], metric = "gower")
sil = silhouette(as.integer(data$kproto_cluster), gowerD)
cat("\nAverage silhouette (Gower):", round(mean(sil[, 3]), 3), "\n")
plot(sil, border = NA, main = "Silhouette (Gower) for K-Prototypes")

if (length(numeric_cols) >= 2) {
  pca = prcomp(scale(data[numeric_cols]), center = FALSE, scale. = FALSE)
  pca_df = data.frame(PC1 = pca$x[,1], PC2 = pca$x[,2], Cluster = data$kproto_cluster)
  ggplot(pca_df, aes(PC1, PC2, color = Cluster)) + geom_point(alpha = 0.7) +
    theme_minimal() + labs(title = "K-Prototypes clusters (PCA of numeric subset)")
}

profile_clusters = function(df, cluster_col, num_cols, cat_cols, top_n = 5) {
  for (cl in levels(df[[cluster_col]])) {
    cat("\n====================\n")
    cat("Cluster", cl, " | Size:", sum(df[[cluster_col]] == cl), "\n")
    cat("--------------------\n")
    sub = df[df[[cluster_col]] == cl, , drop = FALSE]
    if (length(num_cols) > 0) {
      cat("Numeric means (rounded):\n")
      print(round(sapply(sub[num_cols], mean, na.rm = TRUE), 2))
    }
    if (length(cat_cols) > 0) {
      cat("\nTop categorical levels:\n")
      for (cc in cat_cols) {
        freq = sort(prop.table(table(sub[[cc]])), decreasing = TRUE)
        show = head(freq, min(top_n, length(freq)))
        cat(sprintf(" - %s: %s\n", cc, paste(sprintf("%s (%.1f%%)", names(show), 100*as.numeric(show)), collapse = ", ")))
      }
    }
  }
}
profile_clusters(data, "kproto_cluster", numeric_cols, categorical_cols)

if (!is.null(target)) {
  comp = cbind(target, cluster = data$kproto_cluster)
  cat("\nCross-tab: cluster vs target\n")
  print(table(comp$cluster, comp[[1]]))
}

out = if (!is.null(target)) cbind(data, target) else data
write.csv(out, "kproto_clustered_data.csv", row.names = FALSE)


train_df = data1
# Calculate means and standard deviations from training data (before scaling)
train_means = sapply(train_df[numeric_cols], mean, na.rm = TRUE)
train_sds = sapply(train_df[numeric_cols], sd, na.rm = TRUE)

cat("\nTraining data statistics:\n")
cat("Means:\n"); print(round(train_means, 4))
cat("SDs:\n"); print(round(train_sds, 4))

test = read.csv("C:/Users/sheno/OneDrive/Desktop/St4052/Final/Dataset/test.csv")
test=test[,-1]
ytest=test[,21]
# Prepare test data
test_df = test[,-21]

# Convert categorical columns to factors in test data
test_df[categorical_cols] = as.data.frame(
  lapply(test_df[categorical_cols], function(x) {
    if (is.factor(x)) x else factor(x)
  })
)

# Convert numeric columns to numeric type
test_df[numeric_cols] = as.data.frame(
  lapply(test_df[numeric_cols], function(x) {
    as.numeric(as.character(x))
  })
)

# Scale numeric columns using training data means and SDs
test_scaled = test_df
for (col in numeric_cols) {
  test_scaled[[col]] = (test_df[[col]] - train_means[col]) / train_sds[col]
}

cat("\nTest data scaled successfully!\n")
cat("Dimensions:", nrow(test_scaled), "rows x", ncol(test_scaled), "columns\n")

# Preview scaled test data
cat("\nFirst few rows of scaled test data:\n")
print(head(test_scaled))

# Optionally: Predict clusters for test data using the trained k-prototypes model
set.seed(123)
test_clusters = predict(kmod, test_scaled, k = k_opt)

test_df$kproto_cluster = factor(test_clusters$cluster)

cat("\nTest set cluster assignments:\n")
print(table(test_df$kproto_cluster))
test_final=data.frame(test_df,ytest)
# Save test data with cluster assignments
write.csv(test_final, "test_with_clusters.csv", row.names = FALSE)
cat("\nTest data with clusters saved to 'test_with_clusters.csv'\n")



