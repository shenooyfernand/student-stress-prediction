rm(list = ls())

suppressPackageStartupMessages({
  library(cluster)    # silhouette
  library(NbClust)
  library(fpc)        # calinhara, plotcluster
  library(MASS)       # parcoord
  library(mclust)     # adjustedRandIndex
  library(factoextra) # dendrogram viz
})

set.seed(123)

#--------------------------- Load & preprocess ---------------------------
data1 <- read.csv("C:/Users/sheno/OneDrive/Desktop/uni docs/cs assignment/Whitewine.csv")

# keep first 11 predictors, drop duplicate rows, remove NAs, scale
dat_raw <- data1[!duplicated(data1), 1:12]        # assume col 12 is quality (response)
dat_raw <- na.omit(dat_raw)

X <- scale(dat_raw[, 1:11])                        # predictors only
x <- X[, 1:10, drop = FALSE]
y <- dat_raw[, 12]                                 # response for *external* validation only

n <- nrow(x)
max_k <- min(15, n)

#--------------------------- Choose K (NbClust + Elbow) ------------------
# For k-means the 'distance' arg is ignored; k-means uses (squared) Euclidean internally.
set.seed(123)
nb <- NbClust(x, min.nc = 2, max.nc = max_k, method = "kmeans", index = "all")

barplot(table(nb$Best.n[1,]),
        xlab = "Number of Clusters",
        ylab = "Number of Criteria",
        main = "Number of Clusters Chosen by 30 Criteria")

# Elbow (tot.withinss) with robust multiple starts
ks <- 1:max_k
set.seed(123)
wss <- sapply(ks, function(k) kmeans(x, centers = k, nstart = 25)$tot.withinss)
plot(ks, wss, type = "b", xlab = "Number of Clusters", ylab = "Total within-cluster SS",
     main = "Elbow Plot")

#--------------------------- Fit candidates: k = 2 and k = 4 (or 5) ---------------------------
# If your earlier exploration showed k=5 is also plausible, keep both 4 and 5.
set.seed(123)
km2 <- kmeans(x, centers = 2, nstart = 50)
km4 <- kmeans(x, centers = 4, nstart = 50)
# km5 <- kmeans(x, centers = 5, nstart = 50)  # uncomment if you wish to compare 5 as well

# Centers on the scaled space
km2$centers
km4$centers

# Quick visuals
plotcluster(x, km2$cluster); title("k = 2")
parcoord(x, col = km2$cluster, main = "Parallel Coordinates (scaled), k = 2")

plotcluster(x, km4$cluster); title("k = 4")
parcoord(x, col = km4$cluster, main = "Parallel Coordinates (scaled), k = 4")

#--------------------------- Internal validation -------------------------
# Average silhouette (higher is better)
sil2 <- silhouette(km2$cluster, dist(x)); avg_sil2 <- mean(sil2[, 3])
sil4 <- silhouette(km4$cluster, dist(x)); avg_sil4 <- mean(sil4[, 3])

# Calinski–Harabasz (higher is better)
ch2 <- calinhara(x, km2$cluster, cn = 2)
ch4 <- calinhara(x, km4$cluster, cn = 4)

# Davies–Bouldin (lower is better) via clusterSim would be ideal; here rely on sil + CH + elbow.

data.frame(
  k = c(2, 4),
  avg_silhouette = c(avg_sil2, avg_sil4),
  calinski_harabasz = c(ch2, ch4),
  between_over_total = c(km2$betweenss/km2$totss, km4$betweenss/km4$totss)
)

#--------------------------- External validation vs column 12 ---------------------------
# Treat y as labels (if it is numeric quality, ARI still works as it only uses partitions)
ari2 <- adjustedRandIndex(km2$cluster, y)
ari4 <- adjustedRandIndex(km4$cluster, y)

cbind(k = c(2, 4), ARI = c(ari2, ari4))

# Cross-tabs
table_2 <- table(cluster = km2$cluster, quality = y)
table_4 <- table(cluster = km4$cluster, quality = y)
table_2
table_4

#--------------------------- Cluster means on ORIGINAL scale ---------------------------
# Easier to interpret than scaled centers
means_k2 <- aggregate(dat_raw[, 1:11], list(cluster = km2$cluster), mean)
means_k4 <- aggregate(dat_raw[, 1:11], list(cluster = km4$cluster), mean)

means_k2
means_k4

#--------------------------- Hierarchical clustering (improved) -------------------------
# Complete linkage on raw high-dim data tends to give 1 huge cluster + singletons.
# Use Ward.D2 and a PCA reduction to 5 PCs for a balanced result.

pca <- prcomp(x, scale. = FALSE)       # x is already scaled
X5  <- pca$x[, 1:5, drop = FALSE]

hc <- hclust(dist(X5), method = "ward.D2")
plot(hc, main = "Hierarchical Clustering (Ward.D2 on first 5 PCs)")
rect.hclust(hc, k = 4, border = "red")

cut2 <- cutree(hc, k = 2)
cut4 <- cutree(hc, k = 4)

table(cut2)
table(cut4)

fviz_dend(hc, k = 4, cex = 0.4,
          k_colors = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
          color_labels_by_k = TRUE, rect = TRUE)
