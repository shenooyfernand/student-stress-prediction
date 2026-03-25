rm(list=ls())
library(dplyr)
library(ggplot2)
library(GGally)
library(gridExtra)
library(FactoMineR)
library(factoextra)
library(MASS)

path <- "C:/Users/sheno/OneDrive/Desktop/St4052/Final/Dataset/New_cluster_train.csv"
train <- read.csv(path)

num_vars <- c("anxiety_level","blood_pressure","self_esteem")
num_df <- train[, num_vars, drop=FALSE]

iqr_outliers <- function(x){
  q1 <- quantile(x, 0.25, na.rm=TRUE)
  q3 <- quantile(x, 0.75, na.rm=TRUE)
  iqr <- q3 - q1
  lower <- q1 - 1.5*iqr
  upper <- q3 + 1.5*iqr
  which(x < lower | x > upper)
}

uni_idx_list <- lapply(num_df, iqr_outliers)
uni_counts <- sapply(uni_idx_list, length)
uni_df <- data.frame(Variable=names(uni_counts),
                     IQR_Outliers=uni_counts,
                     Percent=round(100*uni_counts/nrow(num_df),2))
print(uni_df)

z_counts <- sapply(num_df, function(x) sum(abs(scale(x))>3, na.rm=TRUE))
z_df <- data.frame(Variable=names(z_counts), Z_Outliers=z_counts)
print(z_df)

plots <- lapply(names(num_df), function(v){
  ggplot(num_df, aes_string(y=v)) +
    geom_boxplot(fill="#1f77b4", color="black", outlier.color="red") +
    theme_minimal(base_size=12) + labs(title=paste("Boxplot –", v))
})
do.call(grid.arrange, c(plots, ncol=3))

ggpairs(num_df,
        upper=list(continuous=wrap("points", alpha=0.6, size=1.5, color="#1f77b4")),
        lower=list(continuous=wrap("smooth", alpha=0.3, color="red")),
        title="Bivariate relationships")

cov_safe <- tryCatch(cov(num_df, use="pairwise.complete.obs"),
                     error=function(e) diag(ncol(num_df)))
center <- colMeans(num_df, na.rm=TRUE)
md <- mahalanobis(num_df, center=center, cov=cov_safe)
md_cut <- qchisq(0.975, df=ncol(num_df))
md_idx <- which(md > md_cut)
cat("Mahalanobis cutoff:", round(md_cut,2), "  N outliers:", length(md_idx), "\n")

ggplot(data.frame(ID=seq_along(md), MD=md), aes(ID, MD)) +
  geom_point(color=ifelse(md>md_cut,"red","black")) +
  geom_hline(yintercept=md_cut, linetype="dashed", color="blue") +
  theme_minimal(base_size=13) + labs(title="Mahalanobis Distance", x="Index", y="MD")

is_num <- names(train) %in% num_vars
cat_vars <- setdiff(names(train), names(train)[is_num])
train[, cat_vars] <- lapply(train[, cat_vars, drop=FALSE], function(x) if(is.factor(x)) x else as.factor(x))
famd_in <- train[, c(cat_vars, num_vars), drop=FALSE]

famd_res <- FAMD(famd_in, graph=FALSE)
coords <- as.data.frame(famd_res$ind$coord)
famd_dist <- sqrt(rowSums(coords^2))
famd_cut <- quantile(famd_dist, 0.975, na.rm=TRUE)
famd_idx <- which(famd_dist > famd_cut)
cat("FAMD cutoff:", round(famd_cut,3), "  N outliers:", length(famd_idx), "\n")

ggplot(data.frame(ID=1:nrow(famd_in), Dist=famd_dist), aes(ID, Dist)) +
  geom_point(color=ifelse(famd_dist>famd_cut,"red","black")) +
  geom_hline(yintercept=famd_cut, linetype="dashed", color="blue") +
  theme_minimal(base_size=13) + labs(title="FAMD Distance", x="Index", y="Distance")

n <- nrow(train)
uni_flag <- rep(FALSE, n); z_flag <- rep(FALSE, n)
for(v in names(uni_idx_list)) uni_flag[uni_idx_list[[v]]] <- TRUE
for(v in names(num_df)) z_flag[which(abs(scale(num_df[[v]]))>3)] <- TRUE
md_flag <- rep(FALSE, n); md_flag[md_idx] <- TRUE
famd_flag <- rep(FALSE, n); famd_flag[famd_idx] <- TRUE

summary_tab <- data.frame(
  IQR_Outlier=sum(uni_flag),
  Z_Outlier=sum(z_flag, na.rm=TRUE),
  MD_Outlier=sum(md_flag),
  FAMD_Outlier=sum(famd_flag),
  Any_Classic=sum(uni_flag | z_flag | md_flag),
  Any_Mixed=sum(famd_flag),
  Any_All=sum(uni_flag | z_flag | md_flag | famd_flag)
)
print(summary_tab)

flags <- data.frame(
  Row=1:n,
  IQR=uni_flag,
  Z=z_flag,
  MD=md_flag,
  FAMD=famd_flag,
  Any=uni_flag | z_flag | md_flag | famd_flag
)
head(flags[order(-as.integer(flags$Any)),], 20)

top_md <- head(order(md, decreasing=TRUE), 10)
top_famd <- head(order(famd_dist, decreasing=TRUE), 10)
cat("Top MD rows:", paste(top_md, collapse=", "), "\n")
cat("Top FAMD rows:", paste(top_famd, collapse=", "), "\n")
print(train[top_md, c(num_vars, cat_vars), drop=FALSE])
print(train[top_famd, c(num_vars, cat_vars), drop=FALSE])

winsorize <- function(x, p=0.01){
  lo <- quantile(x, p, na.rm=TRUE); hi <- quantile(x, 1-p, na.rm=TRUE)
  pmin(pmax(x, lo), hi)
}
train_wins <- train
for(v in num_vars) train_wins[[v]] <- winsorize(train[[v]], p=0.01)

keep_idx <- which(!(md_flag | famd_flag))
train_clean <- train[keep_idx, , drop=FALSE]

out_dir <- dirname(path)

