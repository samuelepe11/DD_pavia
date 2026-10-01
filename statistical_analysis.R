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
                           "Familiar residents", "Unfamiliar residents", "Fast evaluators", "Slow evaluators", "AI wrong", "AI right")

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
      theme(plot.title = element_text(hjust = 0.5), panel.grid = element_blank(), axis.title = element_blank(), axis.text = element_blank(),
        axis.ticks = element_blank())
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
      results <- results %>% mutate(Overreliance = if_else(Model_correct == 0, as.numeric(Diagnosis == Model_pred), NA_real_))
    } else if (var_name == "Underreliance") {
      results <- results %>% mutate(Underreliance = if_else(Model_correct == 1, as.numeric(Diagnosis != Model_pred), NA_real_))
    }
  } else {
    if (var_name == "Accuracy") {
      results <- results %>% mutate(Accuracy = Score / 100)
    } else if (var_name == "Compliance") {
      results <- results %>% mutate(Compliance = 1 - abs(Score - Model_score) / 100)
    } else if (var_name == "Overreliance") {
      results <- results %>% mutate(Overreliance = if_else(Model_correct == 0, 1 - abs(Score - Model_score) / 100, NA_real_))
    } else if (var_name == "Underreliance") {
      results <- results %>% mutate(Underreliance = if_else(Model_correct == 1, abs(Score - Model_score) / 100, NA_real_))
    }
  }
  summary <- results %>% group_by(Annotator, Group) %>% summarise(!!var_name := mean(.data[[var_name]], na.rm = TRUE), .groups = "drop")
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
    norm1 <- ifelse(length(unique(x1)) >= 3 | length(unique(x2)) >= 3, FALSE, shapiro.test(x2 - x1)$p.value > alpha)
    norm2 <- norm1
    same_var <- FALSE
  }
  return(list(norm1 = norm1, norm2 = norm2, same_var = same_var))
}

# Compare statistics
stat_compare <- function(results, var_name, groups, h1 = "greater", alpha = 0.05, is_ordinal = FALSE, is_paired = FALSE, 
                         random_mu = 2, results2=NULL) {
  
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
  return(list(res1 = results1, res2 = results2))
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
      labs(title = paste(toupper(var_name), "COMPARISON BARPLOT"), x = "Group", y = var_name) + coord_cartesian(ylim = ylims)
    
    if (!is.null(mu))
      barplot <- barplot + geom_hline(yintercept = mu, linetype = "dashed", color = "purple")
    
    if (!is.null(levels)) {
      boundaries <- cumsum(group_sizes) + 0.5
      barplot <- barplot + theme(axis.text.x = element_text(angle = 20, hjust = 1)) + 
        geom_vline(xintercept = boundaries, linetype = "dashed", color = "black", alpha = 0.5)
    }
    ggsave(paste0(results_dir, "barplot_", tolower(var_name), title_addon,  ".png"), plot = barplot, dpi = 300, width = width, 
           height = 4)
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
  ggsave(paste0(results_dir, "boxplot_", tolower(var_name), title_addon, ".png"), plot = boxplot, dpi = 300, width = width, height = 4)
  
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
      theme(legend.position = "none", plot.title = element_text(hjust = 0.5, face = "bold", size = 14), axis.text.x = element_text(angle = 15, hjust = 1),
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
        preliminary_questionnaire[[length(preliminary_questionnaire) + 1]] <- read.csv(paste0(folder, "/preliminary_questionnaire.csv"))
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
preliminary_questionnaire$career <- factor(preliminary_questionnaire$career, levels = career_levels, labels = career_levels_en)
preliminary_questionnaire$expertise <- ifelse(preliminary_questionnaire$expertise >= 500, "Expert", "Non-expert")
preliminary_questionnaire$expertise <- factor(preliminary_questionnaire$expertise, levels = c("Non-expert", "Expert"))

survey_results$annotator <- factor(survey_results$annotator, levels = annotator_levels)
survey_results$instance <- factor(survey_results$instance, levels = data_names)
survey_results$visual_group <- factor(survey_results$group, levels = visual_groups, labels = visual_groups_en)
survey_results$group <- ifelse(survey_results$visual_group == visual_groups_en[1], groups[1], 
                               ifelse(survey_results$cam == cams[1], groups[2], groups[3]))

final_questionnaire$annotator <- factor(final_questionnaire$annotator, levels = annotator_levels)
final_questionnaire$visual_group <- factor(final_questionnaire$group, levels = visual_groups, labels = visual_groups_en)

# Compute auxiliary variables
preliminary_questionnaire$familiarity <- preliminary_questionnaire$knowledge == 1 | preliminary_questionnaire$worked == 1
preliminary_questionnaire$trust <- rowMeans(preliminary_questionnaire[, c("performance", "productivity", "effectiveness")], 
                                            na.rm = TRUE) == 1.0

final_questionnaire$competence <- rowMeans(final_questionnaire[, c("competence1", "competence4")], na.rm = TRUE)
final_questionnaire$autonomy <- rowMeans(final_questionnaire[, c("autonomy2", "autonomy3")], na.rm = TRUE)
final_questionnaire$relatedness <- rowMeans(final_questionnaire[, c("relatedness1", "relatedness2")], na.rm = TRUE)
colnames(final_questionnaire)[colnames(final_questionnaire) %in% c("enjoyment1", "demand1")] <- c("enjoyment", "demand")

# Complement survey data frame with additional information
survey_results <- survey_results %>% left_join(preliminary_questionnaire %>% select(annotator, expertise, familiarity, trust), by = "annotator")
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
survey_results <- survey_results %>% mutate(visual_group = as.character(visual_group)) %>%
  select(-any_of(c("block", "block_start", "LagTime", "Duration"))) %>% left_join(block_order, by = c("annotator", "visual_group"))
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
  survey_results <- survey_results %>% left_join(score_results %>% select(Annotator, Instance, Score), by = c("Annotator", "Instance"))
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
  geom_vline(xintercept = median(preliminary_questionnaire$Age, na.rm = TRUE), linetype = "dashed", color = "red", linewidth = 0.8) +
  scale_x_continuous(breaks = scales::breaks_width(2)) +
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

# Compare accuracy
var_name = "Accuracy"
accuracy_summary <- summarise_data(survey_results, var_name)
cat(paste0("\n", toupper(var_name), " EVALUATION...\n"))
stats_a <- stat_compare(accuracy_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "two.sided")
stats_b <- stat_compare(accuracy_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "two.sided")
stats_c <- stat_compare(accuracy_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "two.sided")
plot_compare(rbind(stats_a$res1, stats_a$res2, stats_b$res2), var_name, do_violin = TRUE)

# Compare compliance
cat("\n---------------------------------------------------------------------------------------------------------------------\n")
var_name = "Compliance"
compliance_summary <- summarise_data(survey_results, var_name)
cat(paste0("\n", toupper(var_name), " EVALUATION...\n"))
stats_a <- stat_compare(compliance_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "two.sided")
stats_b <- stat_compare(compliance_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "two.sided")
stats_c <- stat_compare(compliance_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "two.sided")
plot_compare(rbind(stats_a$res1, stats_a$res2, stats_b$res2), var_name, do_violin = TRUE)

# Compare over-reliance
cat("\n---------------------------------------------------------------------------------------------------------------------\n")
var_name = "Overreliance"
overreliance_summary <- summarise_data(survey_results, var_name)
cat(paste0("\n", toupper(var_name), " EVALUATION...\n"))
stats_a <- stat_compare(overreliance_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "two.sided")
stats_b <- stat_compare(overreliance_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "two.sided")
stats_c <- stat_compare(overreliance_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "two.sided")
plot_compare(rbind(stats_a$res1, stats_a$res2, stats_b$res2), var_name, do_violin = TRUE)

# Compare under-reliance
cat("\n---------------------------------------------------------------------------------------------------------------------\n")
var_name = "Underreliance"
underreliance_summary <- summarise_data(survey_results, var_name)
cat(paste0("\n", toupper(var_name), " EVALUATION...\n"))
stats_a <- stat_compare(underreliance_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "two.sided")
stats_b <- stat_compare(underreliance_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "two.sided")
stats_c <- stat_compare(underreliance_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "two.sided")
plot_compare(rbind(stats_a$res1, stats_a$res2, stats_b$res2), var_name, do_violin = TRUE)

# Compare confidence
cat("\n---------------------------------------------------------------------------------------------------------------------\n")
var_name = "Confidence"
confidence_summary <- summarise_data(survey_results, var_name)
cat(paste0("\n", toupper(var_name), " EVALUATION...\n"))
stats_a <- stat_compare(confidence_summary, var_name, groups = groups[1], h1 = "two.sided", random_mu = 2.5)
stats_a <- stat_compare(confidence_summary, var_name, groups = groups[1], h1 = "greater", random_mu = 2.5)
stats_b <- stat_compare(confidence_summary, var_name, groups = groups[2], h1 = "two.sided", random_mu = 2.5)
stats_b <- stat_compare(confidence_summary, var_name, groups = groups[2], h1 = "greater", random_mu = 2.5)
stats_c <- stat_compare(confidence_summary, var_name, groups = groups[3], h1 = "two.sided", random_mu = 2.5)
stats_c <- stat_compare(confidence_summary, var_name, groups = groups[3], h1 = "greater", random_mu = 2.5)
stats_a <- stat_compare(confidence_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "two.sided")
stats_b <- stat_compare(confidence_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "two.sided")
stats_c <- stat_compare(confidence_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "two.sided")
plot_compare(rbind(stats_a$res1, stats_a$res2, stats_b$res2), var_name, mu = 2.5, do_violin = TRUE)

# Compare usefulness
cat("\n---------------------------------------------------------------------------------------------------------------------\n")
var_name = "Usefulness"
usefulness_summary <- summarise_data(survey_results, var_name)
cat(paste0("\n", toupper(var_name), " EVALUATION...\n"))
stats_a <- stat_compare(usefulness_summary, var_name, groups = groups[1], h1 = "two.sided", random_mu = 2.5)
stats_a <- stat_compare(usefulness_summary, var_name, groups = groups[1], h1 = "greater", random_mu = 2.5)
stats_b <- stat_compare(usefulness_summary, var_name, groups = groups[2], h1 = "two.sided", random_mu = 2.5)
stats_b <- stat_compare(usefulness_summary, var_name, groups = groups[2], h1 = "greater", random_mu = 2.5)
stats_c <- stat_compare(usefulness_summary, var_name, groups = groups[3], h1 = "two.sided", random_mu = 2.5)
stats_c <- stat_compare(usefulness_summary, var_name, groups = groups[3], h1 = "greater", random_mu = 2.5)
stats_a <- stat_compare(usefulness_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "two.sided")
stats_b <- stat_compare(usefulness_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "two.sided")
stats_b <- stat_compare(usefulness_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "greater")
stats_c <- stat_compare(usefulness_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "two.sided")
plot_compare(rbind(stats_a$res1, stats_a$res2, stats_b$res2), var_name, mu = 2.5, do_violin = TRUE)

# Compare perceived complexity
cat("\n---------------------------------------------------------------------------------------------------------------------\n")
var_name = "Complexity"
usefulness_summary <- summarise_data(survey_results, var_name)
cat(paste0("\n", toupper(var_name), " EVALUATION...\n"))
stats_a <- stat_compare(usefulness_summary, var_name, groups = groups[1], h1 = "two.sided", random_mu = 2.5)
stats_a <- stat_compare(usefulness_summary, var_name, groups = groups[1], h1 = "greater", random_mu = 2.5)
stats_b <- stat_compare(usefulness_summary, var_name, groups = groups[2], h1 = "two.sided", random_mu = 2.5)
stats_b <- stat_compare(usefulness_summary, var_name, groups = groups[2], h1 = "greater", random_mu = 2.5)
stats_c <- stat_compare(usefulness_summary, var_name, groups = groups[3], h1 = "two.sided", random_mu = 2.5)
stats_c <- stat_compare(usefulness_summary, var_name, groups = groups[3], h1 = "greater", random_mu = 2.5)
stats_a <- stat_compare(usefulness_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "two.sided")
stats_b <- stat_compare(usefulness_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "two.sided")
stats_b <- stat_compare(usefulness_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "greater")
stats_c <- stat_compare(usefulness_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "two.sided")
plot_compare(rbind(stats_a$res1, stats_a$res2, stats_b$res2), var_name, mu = 2.5, do_violin = TRUE)

# Time comparison
cat("\n---------------------------------------------------------------------------------------------------------------------\n")
var_name = "Duration"
time_summary <- summarise_data(filter_outliers(survey_results, var_name), var_name)
cat(paste0("\n", toupper(var_name), " EVALUATION...\n"))
stats_a <- stat_compare(time_summary, var_name, groups =  c(groups[1], groups[2]), is_paired = TRUE, h1 = "two.sided")
stats_b <- stat_compare(time_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "two.sided")
stats_c <- stat_compare(time_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "two.sided")
plot_compare(rbind(stats_a$res1, stats_a$res2, stats_b$res2), var_name, do_violin = TRUE)

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

# Compare accuracy
cat("\n=====================================================================================================================\n")
var_name = "Accuracy"
accuracy_summary <- summarise_data(survey_results, var_name, is_micro = TRUE)
cat(paste0("\n", toupper(var_name), " MICRO EVALUATION...\n"))
stats_a <- stat_compare(accuracy_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "two.sided")
stats_a <- stat_compare(accuracy_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "greater")
stats_b <- stat_compare(accuracy_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "two.sided")
stats_c <- stat_compare(accuracy_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "two.sided")
stats_c <- stat_compare(accuracy_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "greater")
plot_compare(rbind(stats_a$res1, stats_a$res2, stats_b$res2), var_name, title_addon = "_micro", do_violin = TRUE)

# Compare compliance
cat("\n---------------------------------------------------------------------------------------------------------------------\n")
var_name = "Compliance"
compliance_summary <- summarise_data(survey_results, var_name, is_micro = TRUE)
cat(paste0("\n", toupper(var_name), " MICRO EVALUATION...\n"))
stats_a <- stat_compare(compliance_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "two.sided")
stats_a <- stat_compare(compliance_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "greater")
stats_b <- stat_compare(compliance_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "two.sided")
stats_c <- stat_compare(compliance_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "two.sided")
plot_compare(rbind(stats_a$res1, stats_a$res2, stats_b$res2), var_name, title_addon = "_micro", do_violin = TRUE)

# Compare over-reliance
cat("\n---------------------------------------------------------------------------------------------------------------------\n")
var_name = "Overreliance"
overreliance_summary <- summarise_data(survey_results, var_name, is_micro = TRUE)
cat(paste0("\n", toupper(var_name), " MICRO EVALUATION...\n"))
stats_a <- stat_compare(overreliance_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "two.sided")
stats_b <- stat_compare(overreliance_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "two.sided")
stats_c <- stat_compare(overreliance_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "two.sided")
plot_compare(rbind(stats_a$res1, stats_a$res2, stats_b$res2), var_name, title_addon = "_micro", do_violin = TRUE)

# Compare under-reliance
cat("\n---------------------------------------------------------------------------------------------------------------------\n")
var_name = "Underreliance"
underreliance_summary <- summarise_data(survey_results, var_name, is_micro = TRUE)
cat(paste0("\n", toupper(var_name), " MICRO EVALUATION...\n"))
stats_a <- stat_compare(underreliance_summary, var_name, groups = c(groups[1], groups[2]), is_paired = TRUE, h1 = "two.sided")
stats_b <- stat_compare(underreliance_summary, var_name, groups = c(groups[1], groups[3]), is_paired = TRUE, h1 = "two.sided")
stats_c <- stat_compare(underreliance_summary, var_name, groups = c(groups[2], groups[3]), is_paired = TRUE, h1 = "two.sided")
plot_compare(rbind(stats_a$res1, stats_a$res2, stats_b$res2), var_name, title_addon = "_micro", do_violin = TRUE)




