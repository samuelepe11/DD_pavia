############################################################################################################################
# SETTINGS 
############################################################################################################################

# Clear workspace
rm(list = ls())

# Set working directory
working_dir <- "C:/Users/samue/OneDrive/Desktop/Files/Dottorato/DonaldDuck_pavia/results/survey/"
setwd(working_dir)

# Import packages
library(readxl)
library(dplyr)
library(tidyr)
library(car)
library(ggplot2)
library(effsize)
library(effectsize)

# Define global variables
data_dir <- "survey_results/"
survey_data_dir <- "../jai_results/"
results_dir <- "survey_analyses/"
BIND_USER_RESULTS <- FALSE
CREATE_SCORE_FILE <- FALSE

# Define variable levels
data_names <- c("474l", "378d", "405l", "281c", "413l", "297l", "170l", "093l", 
                "433d", "308c", "312l", "152l", "150l", "330l", "459d", "413d",
                "338l", "229c", "386l", "123l", "226l", "354d", "174l", "113l")
gt_class <- c(1, 1, 1, 0, 0, 1, 0, 1,
              1, 1, 1, 0, 0, 1, 1, 1,
              1, 0, 1, 0, 0, 1, 0, 0)
gt_complexity <- c(1, 1, 0, 0, 0, 0, 0, 0,
                   0, 1, 1, 0, 0, 0, 0, 0,
                   0, 0, 1, 1, 0, 0, 0, 0)
model_correctness <- c(1, 0, 1, 1, 0, 0, 1, 0,
                       1, 0, 0, 1, 1, 0, 1, 0,
                       1, 1, 1, 0, 0, 0, 1, 0)
model_score <- c(50, 0, 50, 100, 0, 0, 100, 0,
                 50, 0, 0, 100, 100, 0, 33, 0,
                 75, 100, 50, 0, 0, 0, 100, 0)

classes = c("Non ci sono fratture", "Il paziente presenta fratture")
classes_en <- c("No fracture", "Fracture")
sex_levels <- c("Maschio", "Femmina", "Non binario", "Altro", "Preferisco non rispondere")
sex_levels_en <- c("Male", "Female", "Non-binary", "Other", "Prefer not to answer")
career_levels = c("Non ho ancora iniziato", "I anno", "II anno", "III anno", "IV anno", "Ho terminato la specializzazione")
career_levels_en <- c("Not started", "I year", "II year", "III year", "IV year", "Completed")

binary_answers_en <- c("No", "Yes")
likert_choices_en = c("Strongly disagree", "Disagree", "Somewhat disagree", "Somewhat agree", "Agree", "Strongly agree")
confidence_levels_en <- c("Very low", "Low", "Somewhat low", "Somewhat high", "High", "Very high")
visual_groups = c("Supporto bounding-box", "Supporto heatmap (calda)", "Supporto heatmap (fredda)")
visual_groups_en <- c("Bounding-box support", "Heatmap support (warm)", "Heatmap support (cool)")
groups <- c("Traditional support", "Low-level judicial support", "High-level judicial support")
cams <- c("HiResCAM_feature_extractor_features_7_1_conv3_class", "HiResCAM_feature_extractor_features_7_2_conv3_class")

stratification_levels <- c("Simple cases", "Complex cases", "Cases deemed simple", "Cases deemed complex", 
                           "Support deemed unuseful", "Support deemed useful", "Non-experts", "Experts", 
                           "Familiar residents", "Unfamiliar residents", "Fast evaluators", "Slow evaluators", "AI wrong", 
                           "AI right")


############################################################################################################################
# FUNCTIONS
############################################################################################################################

# Function for categorical variables
plot_population <- function(data, var, title) {
  counts <- data %>% transmute(Category = .data[[var]]) %>% filter(!is.na(Category)) %>% count(Category, name = "n") %>%
    mutate(Percentage = 100 * n / sum(n))
  if (nrow(counts) == 2) {
    # Pie chart
    p <- ggplot(counts, aes(x = "", y = n, fill = Category)) +
      geom_col(width = 1, color = "white") + coord_polar(theta = "y") +
      geom_text(aes(label = sprintf("%d (%.1f%%)", n, Percentage)), position = position_stack(vjust = 0.5), size = 4) +
      labs(title = title, fill = NULL) +
      theme(plot.title = element_text(hjust = 0.5), panel.grid = element_blank(), axis.title = element_blank(), 
            axis.text = element_blank(), axis.ticks = element_blank())
    } else {
    # Barplot
    p <- ggplot(counts, aes(x = Category, y = n)) + geom_col(fill = "#4472C4", width = 0.7) +
      geom_text(aes(label = sprintf("%d (%.1f%%)", n, Percentage)), vjust = -0.5, size = 3.7) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
      labs(title = title, x = NULL, y = "Number of participants") +
      theme(axis.text.x = element_text(angle = 25, hjust = 1), panel.grid.major.x = element_blank())
  }
  ggsave(paste0(results_dir, "population_", tolower(var), ".png"), plot = p, dpi = 300, width = 7, height = 5)
  return(p)
}

# Compute data summary
summarise_data <- function(results, var_name, is_micro=FALSE) {
  if (!is_micro) {
    if (var_name == "Accuracy") {
      results <- results %>% mutate(Accuracy = Diagnosis == Gt_class)
    } else if (var_name == "Compliance") {
      results <- results %>% mutate(Compliance = Diagnosis == Model_pred)
    } else if (var_name == "Overreliance") {
      results <- results %>% mutate(Overreliance = if_else(Model_correct == 0, as.numeric(Diagnosis == Model_pred), 
                                                           NA_real_))
    } else if (var_name == "Underreliance") {
      results <- results %>% mutate(Underreliance = if_else(Model_correct == 1, as.numeric(Diagnosis != Model_pred), 
                                                            NA_real_))
    }
  } else {
    if (var_name == "Accuracy") {
      results <- results %>% mutate(Accuracy = Score / 100)
    } else if (var_name == "Compliance") {
      results <- results %>% mutate(Compliance = 1 - abs(Score - Model_score) / 100)
    } else if (var_name == "Overreliance") {
      results <- results %>% mutate(Overreliance = if_else(Model_correct == 0, 1 - abs(Score - Model_score) / 100, 
                                                           NA_real_))
    } else if (var_name == "Underreliance") {
      results <- results %>% mutate(Underreliance = if_else(Model_correct == 1, abs(Score - Model_score) / 100, NA_real_))
    }
  }
  summary <- results %>% group_by(Annotator, Group) %>% summarise(!!var_name := mean(.data[[var_name]], na.rm = TRUE), 
                                                                  .groups = "drop")
  return(summary)
}

# Check normality and compare standard deviations
stat_compare_preliminary <- function(x1, x2, results_pooled, var_name, alpha = 0.05, group_var_name = "Group", 
                                     is_paired = FALSE) {
  if (!is_paired) {
    #  Check normality with Shapiro–Wilk test
    norm1 <- ifelse(length(unique(x1)) < 3, FALSE, shapiro.test(x1)$p.value > alpha)
    
    if (!is.null(x2)) {
      norm2 <- ifelse(length(unique(x2)) < 3, FALSE, shapiro.test(x2)$p.value > alpha)
      
      # Compare standard deviations with Levene test if inputs are Gaussian, Brown–Forsythe test otherwise
      center <- ifelse(norm1 && norm2, mean, median)
      same_var <- leveneTest(as.formula(paste(var_name, "~", group_var_name)), 
                             data = results_pooled, alternative = "two.sided")[1, ]$`Pr(>F)`> alpha
    } else {
      norm2 <- norm1
      same_var <- FALSE
    }
  } else {
    #  Check normality with paired Shapiro–Wilk test
    norm1 <- ifelse(length(unique(x1)) >= 3 | length(unique(x2)) >= 3, shapiro.test(x2 - x1)$p.value > alpha, FALSE)
    norm2 <- norm1
    same_var <- FALSE
  }
  return(list(norm1 = norm1, norm2 = norm2, same_var = same_var))
}

# Compare statistics
stat_compare <- function(results, var_name, groups, h1 = "greater", alpha = 0.05, is_ordinal = FALSE, is_paired = FALSE, 
                         random_mu = 2, results2 = NULL) {
  
  # Prepare data
  if (is.null(results2)) {
    results1 <- results[results$Group == groups[1], ]
  } else {
    results1 <- results
  }
  
  if (is_ordinal)
    results1[[var_name]] <- as.numeric(results1[[var_name]])
  x1 <- results1[[var_name]]
  n1 <- length(x1)
  
  x2 <- NULL
  n2 <- Inf
  if (length(groups) == 2) {
    if (is.null(results2)) {
      results2 <- results[results$Group == groups[2], ]
    }
    
    if (is_ordinal) 
      results2[[var_name]] <- as.numeric(results2[[var_name]])
    x2 <- results2[[var_name]]
    n2 <- length(x2)
    
    if (!is_ordinal) {
      results1["Group"] <- groups[1]
      results2["Group"] <- groups[2]
      results_pooled <- rbind(results1, results2)
    }
  }
  if (!is_ordinal) {
    # Check normality and compare standard deviations
    preliminary <- stat_compare_preliminary(x1, x2, results_pooled, var_name, is_paired = is_paired)
  }
  # Compare means with T-test if inputs are Gaussian, Mann-Whitney U rank test otherwise. If not same variance we choose Welch's 
  # T-test (analog to T-test with Satterhwaite method).
  # If data are paired use paired T-test if difference of inputs is Gaussian, Wilcoxon signed-rank test otherwise.
  if (n1 >= 10 && n2 >= 10 && !is_ordinal && preliminary$norm1 && preliminary$norm2) {
    if (length(groups) == 2) {
      p_value <- t.test(x1, x2, var.equal = preliminary$same_var, alternative = h1, paired = is_paired)$p.value
    } else {
      # Compare against random effect
      p_value <- t.test(x1, mu = random_mu, alternative = h1)$p.value
    }
    h0_true <- p_value > alpha
  } else {
    if (length(groups) == 2) {
      p_value <- wilcox.test(x1, x2, alternative = h1, exact = FALSE, paired = is_paired)$p.value
    } else {
      # Compare against random effect
      p_value <- wilcox.test(x1, mu = random_mu, alternative = h1, exact = FALSE)$p.value
    }
    h0_true <- p_value > alpha
  }
  
  # Show results
  mean1 <- round(mean(x1, na.rm = TRUE), 2)
  mean2 <- ifelse(length(groups) == 2, round(mean(x2, na.rm = TRUE), 2), random_mu)
  if (h1 != "two.sided")
    addon = paste0(ifelse(h0_true, "not ", ""), h1, " than ")
  else addon = paste0(ifelse(h0_true, "", "not "), "equal to ")
  second_term <- ifelse(length(groups) == 2, paste0(groups[2], "'s ", var_name), "random case")
  cat(paste0(groups[1], "'s ", var_name, " (", mean1, ") is ", addon, second_term, " (", mean2, ")  -  with p-value = ", 
             format(p_value, scientific = TRUE, digits = 2), ", n1 = ", length(x1), ", and n2 = ", length(x2), "\n"))
  return(list(res1 = results1, res2 = results2, p_value = p_value))
}

# Draw compare plots
plot_compare <- function(results_pooled, var_name, levels = NULL, width = 8, title_addon = "", 
                         mu = NULL, group_sizes=c(2, 2, 3), do_barplot = FALSE, do_violin = FALSE) {
  if (!is.null(levels)) {
    results_pooled <- do.call(rbind, results_pooled)
    results_pooled$Group <- factor(results_pooled$Group, levels = unique(results_pooled$Group), labels = levels)
  } else {
    results_pooled$Group <- factor(results_pooled$Group, levels = groups)
  }
  
  if (var_name == "Accuracy" | var_name == "Compliance" | var_name == "Overreliance" | var_name == "Underreliance") {
    ylims = c(0, 1)
  } else {
    if (var_name == "Duration") {
      ylims = c(0, 5)
    } else {
      ylims = c(0, 5)
    }
  }
  mid_x <- length(unique(results_pooled$Group)) / 2 + 0.5
  
  # Bar-plot comparison
  if (do_barplot) {
    barplot <- ggplot(results_pooled, aes(x = Group, y = .data[[var_name]], fill = Group)) +
      geom_bar(stat = "summary", fun = "mean", position = "dodge") +
      labs(title = paste(toupper(var_name), "COMPARISON BARPLOT"), x = "Group", y = var_name) + 
      coord_cartesian(ylim = ylims)
    
    if (!is.null(mu))
      barplot <- barplot + geom_hline(yintercept = mu, linetype = "dashed", color = "purple")
    
    if (!is.null(levels)) {
      boundaries <- cumsum(group_sizes) + 0.5
      barplot <- barplot + theme(axis.text.x = element_text(angle = 20, hjust = 1)) + 
        geom_vline(xintercept = boundaries, linetype = "dashed", color = "black", alpha = 0.5)
    }
    ggsave(paste0(results_dir, "barplot_", tolower(var_name), title_addon,  ".png"), plot = barplot, dpi = 300, 
           width = width, height = 4)
  }
  
  # Box-plot comparison
  boxplot <- ggplot(results_pooled, aes(x = Group, y = .data[[var_name]], fill = Group)) + geom_boxplot() + 
    labs(title = paste(toupper(var_name), "COMPARISON BOXPLOT"), x = "Group", y = var_name) + coord_cartesian(ylim = ylims)
  
  if (!is.null(mu))
    boxplot <- boxplot + geom_hline(yintercept = mu, linetype = "dashed", color = "purple")
  
  if (!is.null(levels)) {
    boundaries <- cumsum(group_sizes) + 0.5
    boxplot <- boxplot + theme(axis.text.x = element_text(angle = 20, hjust = 1)) + 
      geom_vline(xintercept = boundaries, linetype = "dashed", color = "black", alpha = 0.5)
  }
  ggsave(paste0(results_dir, "boxplot_", tolower(var_name), title_addon, ".png"), plot = boxplot, dpi = 300, width = width, 
         height = 4)
  
  # Violin-plot comparison
  if (do_violin) {
    y_breaks <- if (ylims[2] == 1) seq(0, 1, by = 0.1) else seq(ylims[1], ylims[2], by = 0.5)
    y_minor_breaks <- if (ylims[2] == 1) seq(0, 1, by = 0.05) else seq(ylims[1], ylims[2], by = 0.25)
    violinplot <- ggplot(results_pooled, aes(x = Group, y = .data[[var_name]], fill = Group)) +
      geom_violin(trim = TRUE, alpha = 0.45, color = "gray30", linewidth = 0.4, na.rm = TRUE) +
      geom_jitter(width = 0.10, height = 0, size = 1.5, alpha = 0.45, color = "gray25", na.rm = TRUE) +
      geom_boxplot(width = 0.12, fill = "white", alpha = 0.85, outlier.shape = NA, linewidth = 0.4, na.rm = TRUE) +
      stat_summary(fun = mean, geom = "point", shape = 23, size = 3.5, fill = "black", color = "white", na.rm = TRUE) +
      labs(title = paste(toupper(var_name), "COMPARISON"), x = "Group", y = var_name) +
      coord_cartesian(ylim = ylims) +
      scale_y_continuous(breaks = y_breaks, minor_breaks = y_minor_breaks) +
      theme(legend.position = "none", plot.title = element_text(hjust = 0.5, face = "bold", size = 14), 
            axis.text.x = element_text(angle = 15, hjust = 1),
        panel.grid.major.x = element_blank(), panel.grid.major.y = element_line(color = "gray80", linewidth = 0.5),
        panel.grid.minor.y = element_line(color = "gray90", linewidth = 0.3), panel.grid.minor.x = element_blank())
    
    if (!is.null(mu))
      violinplot <- violinplot + geom_hline(yintercept = mu, linetype = "dashed", color = "purple", linewidth = 0.8)
    
    if (!is.null(levels)) {
      boundaries <- cumsum(group_sizes) + 0.5
      violinplot <- violinplot + theme(axis.text.x = element_text(angle = 20, hjust = 1)) +
        geom_vline(xintercept = boundaries, linetype = "dashed", color = "black", alpha = 0.5)
    }
    
    ggsave(paste0(results_dir, "violinplot_", tolower(var_name), title_addon, ".png"), plot = violinplot, dpi = 300,
           width = width, height = 5)
  }
}

# Remove outliers
filter_outliers <- function(results, var_name, multiplier = 3) {
  q1 <- quantile(results[[var_name]], 0.25, na.rm = TRUE)
  q3 <- quantile(results[[var_name]], 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - multiplier * iqr
  upper_bound <- q3 + multiplier * iqr
  filtered_results <- results %>% filter(.data[[var_name]] >= lower_bound & .data[[var_name]] <= upper_bound)
  
  return(filtered_results)
}

# Benjamini–Hochberg correction of p-values
adjust_pvalues <- function(stats, comparisons, method = "BH") {
  p_values <- sapply(stats, function(x) x$p_value)
  return(tibble(Comparison = comparisons, PValue = p_values, PAdjusted = p.adjust(p_values, method = method)))
}

# One comparison of two subgroups
one_subgroup_stats_compare <- function(results, var_name, groups, filter1, filter2, alpha = 0.05,
                                       is_ordinal = FALSE, is_paired = FALSE, is_micro = FALSE, h1 = "two.sided") {
  res1 <- summarise_data(results %>% filter(!!filter1), var_name, is_micro = is_micro) %>%
    filter(is.finite(.data[[var_name]]))
  res2 <- summarise_data(results %>% filter(!!filter2), var_name, is_micro = is_micro) %>%
    filter(is.finite(.data[[var_name]]))
  
  # Match participants for paired comparisons
  if (is_paired) {
    matched <- inner_join(res1 %>% select(Annotator, Control = all_of(var_name)),
                          res2 %>% select(Annotator, Tested = all_of(var_name)), by = "Annotator",
                          relationship = "one-to-one")
    res1 <- matched %>% transmute(Annotator, Group = groups[1], !!var_name := Control)
    res2 <- matched %>% transmute(Annotator, Group = groups[2], !!var_name := Tested)
  } else {
    res1$Group <- groups[1]
    res2$Group <- groups[2]
  }
  
  if (nrow(res1) < 3 || nrow(res2) < 3) stop("Insufficient participants for statistical comparison.")
  if (is_paired && all(res1[[var_name]] == res2[[var_name]]))
    return(list(res1 = res1, res2 = res2, p_value = 1))
  
  stats <- stat_compare(rbind(res1, res2), var_name, groups = groups, alpha = alpha, is_ordinal = is_ordinal,
                        is_paired = is_paired, h1 = h1)
  return(stats)
}

# Subgroup comparisons
subgroup_stats_compare <- function(results, group, var_name = "Accuracy", alpha = 0.05, is_ordinal = FALSE,
                                   is_paired = TRUE, title_addon = "", do_violin = FALSE, is_micro = FALSE,
                                   duration_results = survey_results) {
  # Select macro-group and compute participant-level duration
  duration_summary <- duration_results %>% group_by(Annotator) %>%
    summarise(Mean_duration = mean(Duration, na.rm = TRUE), .groups = "drop")
  speed_cutoff <- median(duration_summary$Mean_duration, na.rm = TRUE)
  if (!is.finite(speed_cutoff)) stop("Cannot compute evaluator speed.")
  
  results <- results %>% select(-any_of("Mean_duration")) %>%
    left_join(duration_summary, by = "Annotator", relationship = "many-to-one") %>% filter(Group == group)
  
  # Define common subgroup comparisons
  filters <- rlang::quos("Non-experts" = Expertise == "Non-expert", "Experts" = Expertise == "Expert",
                         "Familiar residents" = Familiarity == TRUE, "Unfamiliar residents" = Familiarity == FALSE,
                         "Fast evaluators" = Mean_duration <= speed_cutoff, "Slow evaluators" = Mean_duration > speed_cutoff)
  
  # Add case-related subgroup comparisons for non-questionnaire variables
  if (!(var_name %in% c("Competence", "Autonomy", "Relatedness", "Enjoyment", "Demand"))) {
    filters <- c(rlang::quos("Simple cases" = Gt_complexity == 0, "Complex cases" = Gt_complexity == 1,
                             "Cases deemed simple" = Complexity <= 2.5, "Cases deemed complex" = Complexity > 2.5,
                             "Support deemed unuseful" = Usefulness <= 2.5, "Support deemed useful" = Usefulness > 2.5),
                 filters,
                 rlang::quos("AI wrong" = Model_correct == 0, "AI right" = Model_correct == 1))
  }
  
  # Run subgroup comparisons
  res_list <- list()
  output_list <- list()
  output_list_names <- c()
  level_list <- c()
  group_sizes <- c()
  
  for (i in seq(1, length(filters), by = 2)) {
    name1 <- names(filters)[i]
    name2 <- names(filters)[i + 1]
    subgroup_is_paired <- is_paired && !(name1 %in% c("Non-experts", "Familiar residents", "Fast evaluators"))
    
    cat(paste0("\nStratify by ", name1, " vs. ", name2, ":\n"))
    
    stats <- tryCatch(
      one_subgroup_stats_compare(results, var_name, c(name1, name2), filters[[i]], filters[[i + 1]],
                                 alpha = alpha, is_ordinal = is_ordinal, is_paired = subgroup_is_paired,
                                 is_micro = is_micro, h1 = "two.sided"),
      error = function(e) {
        cat("Comparison skipped:", conditionMessage(e), "\n")
        NULL
      }
    )
    
    if (!is.null(stats) && is.finite(stats$p_value)) {
      res_list <- c(res_list, list(stats$res1, stats$res2))
      output_list <- c(output_list, list(stats))
      output_list_names <- c(output_list_names, paste0(name1, " vs. ", name2))
      level_list <- c(level_list, name1, name2)
      group_sizes <- c(group_sizes, 2)
    }
  }
  
  # Display results
  if (length(res_list) > 0) {
    plot_compare(res_list, var_name, levels = level_list, width = 10, group_sizes = group_sizes,
                 title_addon = paste0("_", gsub("[^[:alnum:]]+", "_", tolower(group)), "_stratified", title_addon,
                                      ifelse(is_micro, "_micro", "")), do_violin = do_violin)
    
    pvalues <- adjust_pvalues(output_list, output_list_names)
    cat("\nBH-ADJUSTED p-values:\n")
    print(pvalues)
  } else {
    cat("No subgroup comparisons could be estimated.\n")
  }
  
  return(invisible(if (length(output_list) > 0) pvalues else NULL))
}

# Single-variable analysis
analyse_variable <- function(results, var_name, random_mu = NULL, filter_outliers = FALSE, is_micro = FALSE) {
  if (filter_outliers) {
    results <- filter_outliers(results, var_name)
  }
  data_summary <- summarise_data(results, var_name, is_micro = is_micro)
  
  cat("\n=====================================================================================================================\n")
  string_addon <- ifelse(is_micro, " MICRO", "")
  cat(paste0("\n", toupper(var_name), string_addon, " EVALUATION...\n"))
  
  # Group comparison
  stats_a <- stat_compare(data_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "two.sided")
  stats_b <- stat_compare(data_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "two.sided")
  stats_c <- stat_compare(data_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "two.sided")
  stats_list <- list(stats_a, stats_b, stats_c)
  comparison_names <- c(paste0(groups[1], " vs. ", groups[2]), paste0(groups[1], " vs. ", groups[3]), 
                       paste0(groups[2], " vs. ", groups[3]))
  title_addon <- ifelse(is_micro, "_micro", "")
  plot_compare(rbind(stats_a$res1, stats_a$res2, stats_b$res2), var_name, mu = random_mu, do_violin = TRUE, 
               title_addon = title_addon)
  
  # Randomness check
  if (!is.null(random_mu)) {
    stats_d <- stat_compare(data_summary, var_name, groups = groups[1], h1 = "two.sided", random_mu = random_mu)
    stats_e <- stat_compare(data_summary, var_name, groups = groups[2], h1 = "two.sided", random_mu = random_mu)
    stats_f <- stat_compare(data_summary, var_name, groups = groups[3], h1 = "two.sided", random_mu = random_mu)
    stats_list <- c(stats_list, list(stats_d, stats_e, stats_f))
    comparison_names <- c(comparison_names, paste0(groups[1:3], " vs. random"))
  }
  
  # p-values correction
  pvalues <- adjust_pvalues(stats_list, comparison_names)
  cat("\nBH-ADJUSTED p-values:\n")
  print(pvalues)
  
  # Sub-grouop comparison
  for (group in groups[1:3]) {
    cat(paste0("\n\nSUBGROUP STATS COMPARE FOR ", toupper(group), ":"))
    subgroup_stats_compare(results, group, var_name = var_name, is_paired = FALSE, do_violin = TRUE)
  }
}

# Compute Hedges' g with small-sample correction
hedges_delta <- function(x_tested, x_control, is_paired = FALSE) {
  if (is_paired) {
    if (length(x_tested) != length(x_control)) stop("Paired groups must have equal sample sizes.")
    differences <- x_tested - x_control
    df <- length(differences) - 1
    numerator <- mean(differences)
    denominator <- sd(differences)
  } else {
    n1 <- length(x_tested)
    n2 <- length(x_control)
    df <- n1 + n2 - 2
    numerator <- mean(x_tested) - mean(x_control)
    denominator <- sqrt(((n1 - 1) * var(x_tested) + (n2 - 1) * var(x_control)) / df)
  }
  if (df <= 1 || !is.finite(denominator) || denominator == 0) return(NA_real_)
  correction <- exp(lgamma(df / 2) - lgamma((df - 1) / 2) - 0.5 * log(df / 2))
  return(correction * numerator / denominator)
}

# Compute effect size from participant-level summaries
compute_delta <- function(results_control, results_tested, var_name = "Accuracy", is_paired = FALSE, is_micro = FALSE) {
  control_summary <- summarise_data(results_control, var_name, is_micro = is_micro)
  tested_summary <- summarise_data(results_tested, var_name, is_micro = is_micro)
  
  if (is_paired) {
    paired_data <- inner_join(control_summary %>% select(Annotator, Control = all_of(var_name)),
                              tested_summary %>% select(Annotator, Tested = all_of(var_name)), by = "Annotator") %>%
      filter(is.finite(Control), is.finite(Tested))
    x_control <- paired_data$Control
    x_tested <- paired_data$Tested
  } else {
    x_control <- as.numeric(control_summary[[var_name]])
    x_tested <- as.numeric(tested_summary[[var_name]])
    x_control <- x_control[is.finite(x_control)]
    x_tested <- x_tested[is.finite(x_tested)]
  }
  return(hedges_delta(x_tested, x_control, is_paired = is_paired))
}

# Bootstrap BCa confidence interval for effect size
assess_delta <- function(results, groups, var_name = "Accuracy", title = "global", n_iter = 1000,
                         alpha = 0.05, margin_error = 0.2, seed = 111099, is_paired = FALSE, is_micro = TRUE,
                         show_bca_plots = FALSE) {
  # Original statistic
  cat(paste0("\nAssessing effect-size for ", toupper(gsub("_", " ", title)), " (", var_name, "):\n"))
  if (length(groups) != 2 || groups[1] == groups[2]) stop("Specify two different group labels.")
  results_control <- results[results$Group == groups[1], , drop = FALSE]
  results_tested <- results[results$Group == groups[2], , drop = FALSE]
  if (nrow(results_control) == 0 || nrow(results_tested) == 0) stop("One or both groups are empty.")
  
  # Prepare participant-level data
  control_summary <- summarise_data(results_control, var_name, is_micro = is_micro) %>%
    transmute(Annotator, Value = .data[[var_name]]) %>% filter(is.finite(Value))
  tested_summary <- summarise_data(results_tested, var_name, is_micro = is_micro) %>%
    transmute(Annotator, Value = .data[[var_name]]) %>% filter(is.finite(Value))
  
  if (is_paired) {
    boot_data <- inner_join(control_summary %>% rename(Control = Value),
                            tested_summary %>% rename(Tested = Value), by = "Annotator")
    if (nrow(boot_data) < 4) stop("Insufficient matched participants for paired bootstrap.")
  } else {
    boot_data <- data.frame(Value = c(control_summary$Value, tested_summary$Value),
                            Condition = c(rep("Control", nrow(control_summary)), rep("Tested", nrow(tested_summary))))
    if (nrow(control_summary) < 4 || nrow(tested_summary) < 4) stop("Insufficient participants for bootstrap.")
  }
  
  # Bootstrap statistic
  statistic <- function(data, idx) {
    sample_data <- data[idx, , drop = FALSE]
    if (is_paired) {
      x_control <- sample_data$Control
      x_tested <- sample_data$Tested
    } else {
      x_control <- sample_data$Value[sample_data$Condition == "Control"]
      x_tested <- sample_data$Value[sample_data$Condition == "Tested"]
    }
    return(hedges_delta(x_tested, x_control, is_paired = is_paired))
  }
  
  # Bootstrap resampling
  set.seed(seed)
  if (is_paired) {
    boot_results <- boot::boot(boot_data, statistic = statistic, R = n_iter)
  } else {
    boot_results <- boot::boot(boot_data, statistic = statistic, R = n_iter, strata = factor(boot_data$Condition))
  }
  initial_delta <- as.numeric(boot_results$t0)
  boot_deltas <- as.numeric(boot_results$t)
  valid <- is.finite(boot_deltas)
  n_invalid <- sum(!valid)
  
  if (!is.finite(boot_results$t0)) stop("Original effect size is undefined.")
  if (sum(valid) < 100) stop("Too few valid bootstrap replicates to estimate the confidence interval.")
  cat("Valid bootstrap replicates:", sum(valid), "/", n_iter, "\n")
  if (n_invalid > 0) warning("Some bootstrap replicates have undefined effect sizes.")
  
  # BCa confidence interval
  bca <- tryCatch(suppressWarnings(boot::boot.ci(boot_results, conf = 1 - alpha, type = "bca")$bca),
                  error = function(e) NULL)
  ci_method <- "BCa"
  
  if (is.null(bca) || any(!is.finite(bca[4:5]))) {
    ci_method <- "Percentile"
    ci <- quantile(boot_deltas[valid], probs = c(alpha / 2, 1 - alpha / 2), na.rm = TRUE, names = FALSE)
    lower_bca <- as.numeric(ci[1])
    upper_bca <- as.numeric(ci[2])
    warning("BCa unavailable: using percentile CI. If bootstrap replicates were invalid, interpret this interval cautiously.")
  } else {
    lower_bca <- as.numeric(bca[4])
    upper_bca <- as.numeric(bca[5])
  }
  
  # Plot
  if (show_bca_plots) {
    plot_data <- data.frame(boot_deltas = boot_deltas[valid])
    p <- ggplot(plot_data, aes(x = boot_deltas)) + geom_histogram(bins = 50, fill = "skyblue", alpha = 0.7) +
      geom_vline(xintercept = initial_delta, color = "red", linetype = "dashed") +
      geom_vline(xintercept = c(lower_bca, upper_bca), color = "green", linetype = "dashed") +
      geom_vline(xintercept = c(-margin_error, margin_error), color = "black") +
      labs(title = paste0("Effect-size assessment for ", toupper(gsub("_", " ", title))),
           x = "Effect Size (Hedges' g)", y = "Frequency")
    
    title_addon <- paste0(title, ifelse(is_micro, "_micro", ""))
    ggsave(filename = paste0(results_dir, title_addon, "_", tolower(var_name), "_delta.png"), plot = p, dpi = 300,
           width = 5, height = 4)
  }
  
  # Non-inferiority and equivalence assessment
  if (n_invalid == 0) {
    if (lower_bca > -margin_error) cat("The data suggest non-inferiority.\n")
    else cat("The data do NOT conclusively suggest non-inferiority.\n")
    
    if (upper_bca < margin_error) cat("The data suggest non-superiority.\n")
    else cat("The data do NOT conclusively suggest non-superiority.\n")
    
    if (lower_bca > -margin_error && upper_bca < margin_error) cat("The data suggest equivalence.\n")
  } else {
    cat("Non-inferiority/equivalence assessment withheld because some bootstrap effect sizes were undefined.\n")
  }
  
  # Output
  cat("Initial effect-size:", round(initial_delta, 3), "\n")
  cat(ci_method, (1 - alpha) * 100, "% CI:", paste0("(", round(lower_bca, 3), ", ", round(upper_bca, 3), ")"), "\n")
  return(data.frame(delta = initial_delta, lower = lower_bca, upper = upper_bca,
                    ci_method = ci_method, invalid_boot = n_invalid))
}

# Draw Forest plot to compare effect sizes
draw_forest_plot <- function(input_values, input_names, margin_error = 0.2, addon = "", width = 8, height = 4,
                             group_sizes = NULL) {
  df <- do.call(rbind, input_values)
  if (nrow(df) != length(input_names)) stop("Number of input values and names must match.")
  if (anyDuplicated(input_names)) stop("Forest plot labels must be unique.")
  df$Group <- factor(input_names, levels = rev(input_names))
  
  p <- ggplot(df, aes(x = delta, y = Group)) +
    annotate("rect", xmin = -margin_error, xmax = margin_error, ymin = -Inf, ymax = Inf, fill = "darkgray", alpha = 0.5) +
    geom_vline(xintercept = 0, color = "black", linewidth = 1.0, alpha = 0.7) +
    geom_vline(xintercept = c(-margin_error, margin_error), linetype = "dashed", color = "gray25", linewidth = 1.0, alpha = 0.7) +
    geom_point(aes(color = Group), size = 3, na.rm = TRUE) +
    geom_errorbar(aes(xmin = lower, xmax = upper, color = Group), orientation = "y", width = 0.2, linewidth = 1.2, na.rm = TRUE) +
    labs(x = "Effect Size (with bootstrap CI)", y = "") +
    theme(legend.position = "none")
  
  # Indicate unavailable estimates
  if (any(!is.finite(df$delta))) {
    p <- p + geom_text(data = df[!is.finite(df$delta), ], aes(x = -0.95, y = Group, label = "Not estimable"),
                       inherit.aes = FALSE, hjust = 0, color = "gray30", size = 3)
  }
  
  # Separate subgroup categories
  if (!is.null(group_sizes)) {
    boundaries <- head(cumsum(group_sizes), -1) + 0.5
    p <- p + geom_hline(yintercept = boundaries, linetype = "dashed", color = "gray55", linewidth = 0.8) +
      theme(panel.grid.major.y = element_blank(), panel.grid.minor.y = element_blank())
  }
  
  # Set grid
  p <- p + scale_x_continuous(breaks = seq(-1, 1, by = 0.2)) + coord_cartesian(xlim = c(-1, 1)) +
    theme(panel.grid.major.x = element_line(color = "white"), panel.grid.minor.x = element_blank())
  
  addon <- ifelse(addon == "", addon, paste0(addon, "_"))
  ggsave(filename = paste0(results_dir, addon, "delta_forest.png"), plot = p, dpi = 300, width = width, height = height)
}

# Subgroup bootstrap CI assessment for effect size given a filtering expression
one_subgroup_delta <- function(results, groups, var_name, filter_expr, title, n_iter = 1000, alpha = 0.05,
                               margin_error = 0.2, seed = 111099, is_paired = TRUE, is_micro = FALSE) {
  filtered_results <- results %>% filter(!!filter_expr)
  stats <- tryCatch(
    assess_delta(filtered_results, groups = groups, var_name = var_name, title = title, n_iter = n_iter,
                 alpha = alpha, margin_error = margin_error, seed = seed, is_paired = is_paired, is_micro = is_micro),
    error = function(e) {
      cat(paste0("Subgroup ", title, " could not be estimated: ", conditionMessage(e), "\n"))
      data.frame(delta = NA_real_, lower = NA_real_, upper = NA_real_, ci_method = "Unavailable", invalid_boot = NA_integer_)
    }
  )
  return(stats)
}

# Define subgroup comparisons
get_subgroup_filters <- function(var_name, speed_cutoff) {
  filters <- rlang::quos("Non-experts" = Expertise == "Non-expert", "Experts" = Expertise == "Expert",
                         "Familiar residents" = Familiarity == TRUE, "Unfamiliar residents" = Familiarity == FALSE,
                         "Fast evaluators" = Mean_duration <= speed_cutoff, "Slow evaluators" = Mean_duration > speed_cutoff)
  
  if (!(var_name %in% c("Competence", "Autonomy", "Relatedness", "Enjoyment", "Demand"))) {
    filters <- c(rlang::quos("Simple cases" = Gt_complexity == 0, "Complex cases" = Gt_complexity == 1,
                             "Cases deemed simple" = Complexity <= 2.5, "Cases deemed complex" = Complexity > 2.5,
                             "Support deemed unuseful" = Usefulness <= 2.5, "Support deemed useful" = Usefulness > 2.5),
                 filters,
                 rlang::quos("AI wrong" = Model_correct == 0, "AI right" = Model_correct == 1))
  }
  return(filters)
}

# Subgroup bootstrap CI assessment for effect size
subgroup_delta <- function(results, groups, var_name = "Accuracy", n_iter = 1000, alpha = 0.05, margin_error = 0.2,
                           seed = 111099, title = "global", is_paired = TRUE, is_micro = FALSE, width = 10, height = 8) {
  # Define subgroup comparisons
  speed_cutoff <- median(results$Mean_duration, na.rm = TRUE)
  filters <- get_subgroup_filters(var_name, speed_cutoff)
  
  # Run subgroup comparisons
  stats <- vector("list", length(filters))
  names(stats) <- names(filters)
  for (i in seq_along(filters)) {
    subgroup_title <- paste0(title, "_", tolower(gsub("[^[:alnum:]]+", "_", names(filters)[i])))
    stats[[i]] <- one_subgroup_delta(results, groups = groups, var_name = var_name, filter_expr = filters[[i]],
                                     title = subgroup_title, n_iter = n_iter, alpha = alpha, margin_error = margin_error,
                                     seed = seed + i, is_paired = is_paired, is_micro = is_micro)
  }
  # Display results
  addon <- paste0("subgroup_compare_", tolower(var_name), ifelse(is_micro, "_micro", ""), "_", title)
  forest <- draw_forest_plot(stats, names(filters), margin_error = margin_error, addon = addon, width = width, height = height,
                             group_sizes = rep(2, length(filters) / 2))
  return(invisible(stats))
}

# Single-variable effect size analysis
analyse_delta <- function(results, var_name, is_micro = FALSE, subgroup_width = 8, subgroup_height = 8) {
  # Group comparison
  addon <- ifelse(is_micro, "_micro", "")
  stats_a <- assess_delta(results, groups = c(groups[1], groups[2]), var_name = var_name, title = "lowlevel", is_paired = TRUE,
                          is_micro = is_micro)
  stats_b <- assess_delta(results, groups = c(groups[1], groups[3]), var_name = var_name, title = "highlevel", is_paired = TRUE,
                          is_micro = is_micro)
  draw_forest_plot(list(stats_a, stats_b), c(groups[2], groups[3]), addon = paste0("scenario_compare_", tolower(var_name), addon), 
                   width = 10, height = 3)
  
  # Sub-group comparison
  subgroup_delta(results, groups = c(groups[1], groups[2]), title = "lowlevel", var_name = var_name, width = subgroup_width, 
                 height = subgroup_height, is_micro = is_micro, is_paired = TRUE)
  subgroup_delta(results, groups = c(groups[1], groups[3]), title = "highlevel", var_name = var_name, width = subgroup_width, 
                 height = subgroup_height, is_micro = is_micro, is_paired = TRUE)
}


############################################################################################################################
# READ DATA 
############################################################################################################################

# Read user data
if (BIND_USER_RESULTS) {
  # Concatenate user results
  preliminary_questionnaire <- list()
  survey_results <- list()
  final_questionnaire <- list()
  for (folder in list.dirs(data_dir, recursive = FALSE)) {
    csv_files <- list.files(folder)
    if (length(csv_files) == 3 & folder != "survey_results/Pietrasanta" & folder != "survey_results/Paparo" & 
        folder != "survey_results/Francesca Antonuccio" & folder != "survey_results/Boh" & folder != "survey_results/Antonazzi") {
      final_i <- read.csv(paste0(folder, "/final_questionnaire.csv"))
      if (nrow(final_i) == 3) {
        final_questionnaire[[length(final_questionnaire) + 1]] <- final_i
        preliminary_questionnaire[[length(preliminary_questionnaire) + 1]] <- read.csv(paste0(folder, 
                                                                                              "/preliminary_questionnaire.csv"))
        survey_results[[length(survey_results) + 1]] <- read.csv(paste0(folder, "/survey_results.csv"))
      }
      rm(final_i)
    }
  }
  # Concatenate results
  preliminary_questionnaire <- do.call(rbind, preliminary_questionnaire)
  survey_results <- do.call(rbind, survey_results)
  final_questionnaire <- do.call(rbind, final_questionnaire)
  
  # Adjust time for temporal checks and uniformity
  preliminary_questionnaire$time <- as.POSIXct(preliminary_questionnaire$time, format = "%Y-%m-%d %H:%M:%S")
  survey_results$time <- as.POSIXct(survey_results$time, format = "%Y-%m-%d %H:%M:%S")
  final_questionnaire$time <- as.POSIXct(final_questionnaire$time, format = "%Y-%m-%d %H:%M:%S")
  
  # Remove duplicates before saving
  survey_results <- survey_results %>% arrange(desc(time)) %>% distinct(instance, annotator, group, .keep_all = TRUE)
  
  # Save final CSV files
  write.csv(survey_results, paste0(results_dir, "survey_results_all.csv"), row.names = FALSE)
  write.csv(preliminary_questionnaire, paste0(results_dir, "preliminary_questionnaire_all.csv"), row.names = FALSE)
  write.csv(final_questionnaire, paste0(results_dir, "final_questionnaire_all.csv"), row.names = FALSE)
  
} else {
  # Read concatenated files
  preliminary_questionnaire <- read.csv(paste0(results_dir, "preliminary_questionnaire_all.csv"))
  survey_results <- read.csv(paste0(results_dir, "survey_results_all.csv"))
  final_questionnaire <- read.csv(paste0(results_dir, "final_questionnaire_all.csv"))
}


# Adjust variable types
annotator_levels <- preliminary_questionnaire$annotator
preliminary_questionnaire$annotator <- factor(preliminary_questionnaire$annotator, levels = annotator_levels)
preliminary_questionnaire$sex <- factor(preliminary_questionnaire$sex, levels = sex_levels, labels = sex_levels_en)
preliminary_questionnaire$career <- factor(preliminary_questionnaire$career, levels = career_levels, 
                                           labels = career_levels_en)
preliminary_questionnaire$expertise <- ifelse(preliminary_questionnaire$expertise >= 500, "Expert", "Non-expert")
preliminary_questionnaire$expertise <- factor(preliminary_questionnaire$expertise, levels = c("Non-expert", "Expert"))

survey_results$annotator <- factor(survey_results$annotator, levels = annotator_levels)
survey_results$instance <- factor(survey_results$instance, levels = data_names)
survey_results$visual_group <- factor(survey_results$group, levels = visual_groups, labels = visual_groups_en)
survey_results$group <- ifelse(survey_results$visual_group == visual_groups_en[1], groups[1], 
                               ifelse(survey_results$cam == cams[1], groups[2], groups[3]))

final_questionnaire$annotator <- factor(final_questionnaire$annotator, levels = annotator_levels)
final_questionnaire$visual_group <- factor(final_questionnaire$group, levels = visual_groups, labels = visual_groups_en)
group_mapping <- survey_results %>% distinct(annotator, visual_group, group)
final_questionnaire <- final_questionnaire %>% select(-group) %>% left_join(group_mapping, by = c("annotator", "visual_group"))

# Compute auxiliary variables
preliminary_questionnaire$familiarity <- preliminary_questionnaire$knowledge == 1 | preliminary_questionnaire$worked == 1
preliminary_questionnaire$trust <- rowMeans(preliminary_questionnaire[, c("performance", "productivity", "effectiveness")], 
                                            na.rm = TRUE) == 1.0

final_questionnaire$competence <- rowMeans(final_questionnaire[, c("competence1", "competence4")], na.rm = TRUE)
final_questionnaire$autonomy <- rowMeans(final_questionnaire[, c("autonomy2", "autonomy3")], na.rm = TRUE)
final_questionnaire$relatedness <- rowMeans(final_questionnaire[, c("relatedness1", "relatedness2")], na.rm = TRUE)
colnames(final_questionnaire)[colnames(final_questionnaire) %in% c("enjoyment1", "demand1")] <- c("enjoyment", "demand")

# Complement survey data frame with additional information
survey_results <- survey_results %>% left_join(preliminary_questionnaire %>% select(annotator, expertise, 
                                                                                    familiarity, trust), by = "annotator")
survey_results$gt_class <- gt_class[match(as.character(survey_results$instance), data_names)]
survey_results$gt_complexity <- gt_complexity[match(as.character(survey_results$instance), data_names)]

# Insert model predictions
survey_results$model_correct <- model_correctness[match(as.character(survey_results$instance), data_names)]
survey_results$model_pred <- ifelse(survey_results$model_correct, survey_results$gt_class, 1 - survey_results$gt_class)
survey_results$model_score <- model_score[match(as.character(survey_results$instance), data_names)]

# Compute duration
block_order <- final_questionnaire %>% mutate(visual_group = as.character(visual_group)) %>% arrange(annotator, time) %>%
  group_by(annotator) %>% mutate(block = row_number() - 1, block_start = lag(time)) %>% ungroup() %>%
  left_join(preliminary_questionnaire %>% select(annotator, time_pre = time), by = "annotator") %>%
  mutate(block_start = if_else(block == 0, time_pre, block_start)) %>% select(annotator, visual_group, block, block_start)
suppressWarnings(
  survey_results <- survey_results %>% mutate(visual_group = as.character(visual_group)) %>%
    select(-any_of(c("block", "block_start", "LagTime", "Duration"))) %>% left_join(block_order, by = c("annotator",
                                                                                                        "visual_group"))
  )
survey_results <- survey_results %>% group_by(annotator, block) %>% arrange(time, .by_group = TRUE) %>%
  mutate(LagTime = lag(time), LagTime = if_else(row_number() == 1, block_start, LagTime)) %>% ungroup() %>%
  mutate(Duration = as.numeric(difftime(time, LagTime, units = "mins"))) %>% select(-LagTime, -block_start)
rm(block_order)

# Adjust variable names
colnames(preliminary_questionnaire) <- tools::toTitleCase(colnames(preliminary_questionnaire))
colnames(survey_results) <- tools::toTitleCase(colnames(survey_results))
colnames(final_questionnaire) <- tools::toTitleCase(colnames(final_questionnaire))

# Create scoring file
if (CREATE_SCORE_FILE) {
  score_results <- survey_results %>% transmute(Annotator, Instance, Diagnosis, Score = NA_real_, Location)
  write.csv2(score_results, paste0(results_dir, "score_results.csv"), row.names = FALSE, na = "")
} else {
  score_results <- read_excel(paste0(results_dir, "score_results.xlsx"))
  score_results$Annotator[score_results$Annotator == "Ellena"] <- "Ellena "
  survey_results <- survey_results %>% left_join(score_results %>% select(Annotator, Instance, Score), by = c("Annotator", 
                                                                                                              "Instance"))
}


############################################################################################################################
# PRELIMINARY ANALYSIS
############################################################################################################################

# Users' characteristics
summary(preliminary_questionnaire)

# Prepare categorical variables
population <- preliminary_questionnaire %>%
  mutate(Familiarity = factor(Familiarity, levels = c(FALSE, TRUE), labels = c("Non-familiar", "Familiar")),
         Skepticism = factor(Trust, levels = c(FALSE, TRUE), labels = c("Skeptic", "Non-skeptic")))
p_sex <- plot_population(population, "Sex", "Sex Distribution")
p_career <- plot_population(population, "Career", "Specialization Year Distribution")
p_expertise <- plot_population(population, "Expertise", "Clinical Expertise Distribution")
p_familiarity <- plot_population(population, "Familiarity", "Familiarity with AI")
p_skeptic <- plot_population(population, "Skepticism", "Skepticism towards AI")

# Age distribution
p_age <- ggplot(preliminary_questionnaire, aes(x = Age)) +
  geom_histogram(binwidth = 2, boundary = 0, fill = "#4472C4", color = "white") +
  geom_vline(xintercept = median(preliminary_questionnaire$Age, na.rm = TRUE), linetype = "dashed", color = "red", 
             linewidth = 0.8) + scale_x_continuous(breaks = scales::breaks_width(2)) +
  labs(title = "Age Distribution", x = "Age (years)", y = "Number of participants")
ggsave(paste0(results_dir, "population_age.png"), plot = p_age, dpi = 300, width = 7, height = 5)


############################################################################################################################
# STATISTICAL ANALYSES FOR MACRO ACCURACY
############################################################################################################################

# Diagnosis distribution by group
diagnosis_distribution <- survey_results %>%
  mutate(Diagnosis = factor(Diagnosis, levels = c(0, 1), labels = classes_en)) %>%
  count(Group, Diagnosis) %>% group_by(Group) %>% mutate(Proportion = n / sum(n)) %>% ungroup()
p_diagnosis <- ggplot(diagnosis_distribution, aes(x = Group, y = Proportion, fill = Diagnosis)) +
  geom_col(position = "dodge") + scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  labs(title = "Distribution of Selected Diagnoses by Group", x = "Group", y = "Selected diagnoses (%)", 
       fill = "Diagnosis")
ggsave(filename = paste0(results_dir, "diagnosis_distribution_by_group.png"), plot = p_diagnosis, dpi = 300, width = 8, 
       height = 5)

# Compare variables
analyse_variable(survey_results, var_name = "Accuracy")
analyse_variable(survey_results, var_name = "Compliance")
analyse_variable(survey_results, var_name = "Overreliance")
analyse_variable(survey_results, var_name = "Underreliance")
analyse_variable(survey_results, var_name = "Confidence", random_mu = 2.5)
analyse_variable(survey_results, var_name = "Usefulness", random_mu = 2.5)
print(ok)
analyse_variable(survey_results, var_name = "Complexity", random_mu = 2.5)
analyse_variable(survey_results, var_name = "Duration", filter_outliers = TRUE)


############################################################################################################################
# STATISTICAL ANALYSES FOR MICRO ACCURACY
############################################################################################################################

# Score distribution by group
p_score <- ggplot(survey_results, aes(x = Score, fill = Group)) +
  geom_histogram(aes(y = after_stat(count / sum(count))), bins = 10, color = "white", na.rm = TRUE) +
  facet_wrap(~Group, nrow = 1) + scale_y_continuous(labels = scales::label_percent()) + 
  labs(title = "Score Distribution by Group", x = "Score", y = "Observations (%)") +
  theme(legend.position = "none")
ggsave(filename = paste0(results_dir, "score_distribution_by_group.png"), plot = p_score, dpi = 300, width = 12, height = 4)

# Compare variables
analyse_variable(survey_results, var_name = "Accuracy", is_micro = TRUE)
analyse_variable(survey_results, var_name = "Compliance", is_micro = TRUE)
analyse_variable(survey_results, var_name = "Overreliance", is_micro = TRUE)
analyse_variable(survey_results, var_name = "Underreliance", is_micro = TRUE)


############################################################################################################################
# FINAL QUESTIONNAIRE EVALUATION
############################################################################################################################

# Solve data issues
final_questionnaire <- final_questionnaire %>% left_join(preliminary_questionnaire %>% 
                                                           select(Annotator, Expertise, Familiarity, Trust), by = "Annotator")

# Asses preferences
cat("\n=====================================================================================================================\n")
pref_df <- final_questionnaire %>% filter(!is.na(Preference), Preference != "") %>% count(Preference) %>%
  mutate(perc = round(100 * n / sum(n), 1), label = paste0(Preference, "\n", n, " (", perc, "%)"))
pref_df$Preference <- factor(pref_df$Preference, levels = unique(pref_df$Preference)[c(1, 4, 3, 2)])
p_pref <- ggplot(pref_df, aes(x = "", y = n, fill = Preference)) + geom_col(width = 1) + coord_polar(theta = "y") +
  geom_text(aes(label = label), position = position_stack(vjust = 0.5), size = 4) + labs(title = "Preference distribution", 
                                                                                         fill = "Preference") + 
  theme(plot.title = element_text(hjust = 0.5), panel.grid = element_blank(), axis.title = element_blank(), 
        axis.text = element_blank(),axis.ticks = element_blank(), legend.position = "none")
ggsave(filename = paste0(results_dir, "preference_pieplot.png"), plot = p_pref, dpi = 300, width = 8, height = 6)

# Compare variables
analyse_variable(final_questionnaire, var_name = "Competence", random_mu = 2.5)
analyse_variable(final_questionnaire, var_name = "Autonomy", random_mu = 2.5)
analyse_variable(final_questionnaire, var_name = "Relatedness", random_mu = 2.5)
analyse_variable(final_questionnaire, var_name = "Enjoyment", random_mu = 2.5)
analyse_variable(final_questionnaire, var_name = "Demand", random_mu = 2.5)


############################################################################################################################
# EFFECT SIZE EVALUATION
############################################################################################################################

# Compute participant-level evaluation speed
duration_summary <- survey_results %>% group_by(Annotator) %>%
  summarise(Mean_duration = mean(Duration, na.rm = TRUE), .groups = "drop")
survey_results <- survey_results %>% left_join(duration_summary, by = "Annotator")
final_questionnaire <- final_questionnaire %>% left_join(duration_summary, by = "Annotator")

# Macro-measurements
analyse_delta(survey_results, "Accuracy", subgroup_height = 5)
analyse_delta(survey_results, "Compliance", subgroup_height = 5)
analyse_delta(survey_results, "Overreliance", subgroup_height = 5)
analyse_delta(survey_results, "Underreliance", subgroup_height = 5)
analyse_delta(survey_results, "Confidence", subgroup_height = 5)
analyse_delta(survey_results, "Usefulness", subgroup_height = 5)

# Micro-measurements
analyse_delta(survey_results, "Accuracy", is_micro = TRUE, subgroup_height = 5)
analyse_delta(survey_results, "Compliance", is_micro = TRUE, subgroup_height = 5)
analyse_delta(survey_results, "Overreliance", is_micro = TRUE, subgroup_height = 5)
analyse_delta(survey_results, "Underreliance", is_micro = TRUE, subgroup_height = 5)

# Final questionnaire measurements
analyse_delta(final_questionnaire, "Competence", subgroup_height = 4)
analyse_delta(final_questionnaire, "Autonomy", subgroup_height = 4)
analyse_delta(final_questionnaire, "Relatedness", subgroup_height = 4)
analyse_delta(final_questionnaire, "Enjoyment", subgroup_height = 4)
analyse_delta(final_questionnaire, "Demand", subgroup_height = 4)
