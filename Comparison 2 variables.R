rm(list=ls())
data = read.csv("C:/Users/sheno/OneDrive/Desktop/St4052/Final/Dataset/train.csv")
data = data[,-1]
numeric_vars <- c("anxiety_level", "depression", "self_esteem",
                  "blood_pressure")   # add others if needed
cat_vars <- setdiff(names(data), c(numeric_vars, "stress_level"))

# --- Initialize results data frame ---
chi_results <- data.frame(Variable = character(),
                          Chi_square = numeric(),
                          df = numeric(),
                          p_value = numeric(),
                          stringsAsFactors = FALSE)

# --- Run Chi-square test for each categorical variable ---
for (var in cat_vars) {
  tbl <- table(data[[var]], data$stress_level)
  if (all(rowSums(tbl) > 0) && all(colSums(tbl) > 0)) {
    test <- suppressWarnings(chisq.test(tbl))
    chi_results <- rbind(chi_results,
                         data.frame(Variable = var,
                                    Chi_square = round(test$statistic, 3),
                                    df = test$parameter,
                                    p_value = round(test$p.value, 5)))
  }
}

# --- Display results ---
chi_results <- chi_results[order(chi_results$p_value), ]
print(chi_results)

# --- Optional: show only significant results (p < 0.05) ---
sig_results <- subset(chi_results, p_value < 0.05)
print(sig_results)
