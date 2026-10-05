#!/usr/bin/env Rscript

library(dplyr)
library(lme4)
library(emmeans)
library(lmerTest)
library(ggplot2)
library(ggbeeswarm)
library(scales)
library(latex2exp)

cat('\n') # terminal spacing for readability

#
#

# specify names
treatment_name <- "SLK/LOK Inhibitor"
# treatment_name <- "ROCK Inhibitor"
# treatment_name <- "Control +DMSO"

# import data
data <- read.csv("slk_median_CH-areas.csv")
# data <- read.csv("rock_median_CH-areas.csv")
# data <- read.csv("ctrl_median_CH-areas.csv")

# rename columns for formatting
names(data)[names(data) == "Median.Convex.Hull.Area"] <- "CHA"
names(data)[names(data) == "Combined"] <- "Dataset"
names(data)[names(data) == "Location"] <- "Area"

rep0 <- paste(treatment_name, "Rep 0")
rep1 <- paste(treatment_name, "Rep 1")
rep2 <- paste(treatment_name, "Rep 2")

# rename Experiments for plotting homogeneity
data$Experiment <- gsub(rep0, "1", data$Experiment)
data$Experiment <- gsub(rep1, "2", data$Experiment)
data$Experiment <- gsub(rep2, "3", data$Experiment)

# order pre before post; junction before non-junction
data$Dataset	<- factor(data$Dataset, levels = c("Pre, Junction", "Post, Junction", "Pre, Non-junction", "Post, Non-junction"))

# filter data
junc_data 		<- data %>%
	filter(Area == "Junction")
nonjunc_data	<- data %>%
	filter(Area == "Non-junction")
pre_data			<- data %>%
	filter(Treatment == "Pre")
post_data			<- data %>%
	filter(Treatment == "Post")

# order filtered data - pre before post; junction before non-junction
junc_data$Treatment			<- factor(junc_data$Treatment, levels = c("Pre", "Post"))
nonjunc_data$Treatment	<- factor(nonjunc_data$Treatment, levels = c("Pre", "Post"))
pre_data$Area						<- factor(pre_data$Area, levels = c("Junction", "Non-junction"))
post_data$Area					<- factor(post_data$Area, levels = c("Junction", "Non-junction"))

# ensure names are file system safe
if (treatment_name == "SLK/LOK Inhibitor") {
	safe_treatment_name <- "SLK-LOK Inhibitor"
} else {
	safe_treatment_name <- treatment_name
}

if (treatment_name == "Control +DMSO") {
	treatment_name <- "Control \\+DMSO"
}

# fit data for each condition pair to its own model
junc_model 		<- lmer(CHA ~ Treatment + (1 | Experiment), data = junc_data)
nonjunc_model	<- lmer(CHA ~ Treatment + (1 | Experiment), data = nonjunc_data)
pre_model			<- lmer(CHA ~ Area + (1 | Experiment), data = pre_data)
post_model		<- lmer(CHA ~ Area + (1 | Experiment), data = post_data)

# get SEM via emmeans and add "Dataset" column for plotting
junc_emm 			<- emmeans(junc_model, ~ Treatment)
junc_sem_data <- as.data.frame(junc_emm)
junc_sem_data <- junc_sem_data %>% mutate(Dataset = paste0(Treatment, ", Junction"))

nonjunc_emm 			<- emmeans(nonjunc_model, ~ Treatment)
nonjunc_sem_data 	<- as.data.frame(nonjunc_emm)
nonjunc_sem_data  <- nonjunc_sem_data %>% mutate(Dataset = paste0(Treatment, ", Non-junction"))

pre_emm 			<- emmeans(pre_model, ~ Area)
pre_sem_data 	<- as.data.frame(pre_emm)
pre_sem_data  <- pre_sem_data %>% mutate(Dataset = paste0("Pre, ", Area))

post_emm 			<- emmeans(post_model, ~ Area)
post_sem_data <- as.data.frame(post_emm)
post_sem_data <- post_sem_data %>% mutate(Dataset = paste0("Post, ", Area))

### get predicted effect from model ###

# predict fixed effect, ignoring random effects (re.form = NA)
junc_data$Model_Fit			<- predict(junc_model, newdata = junc_data, re.form = NA)
nonjunc_data$Model_Fit	<- predict(nonjunc_model, newdata = nonjunc_data, re.form = NA)
pre_data$Model_Fit			<- predict(pre_model, newdata = pre_data, re.form = NA)
post_data$Model_Fit			<- predict(post_model, newdata = post_data, re.form = NA)

# get model summaries
junc_model_summary 		= summary(junc_model)
nonjunc_model_summary = summary(nonjunc_model)
pre_model_summary 		= summary(pre_model)
post_model_summary 		= summary(post_model)

# get coefficient table
junc_coef_table 			<- coef(junc_model_summary)
nonjunc_coef_table		<- coef(nonjunc_model_summary)
pre_coef_table				<- coef(pre_model_summary)
post_coef_table				<- coef(post_model_summary)

# extract p-values
junc_p_vals			<- junc_coef_table[, "Pr(>|t|)"]
nonjunc_p_vals	<- nonjunc_coef_table[, "Pr(>|t|)"]
pre_p_vals			<- pre_coef_table[, "Pr(>|t|)"]
post_p_vals			<- post_coef_table[, "Pr(>|t|)"]

# get significance codes for treatment effects
cutpoints = c(0, 0.001, 0.01, 0.05, 1)
symbols = c("***", "**", "*", "ns")

junc_sig <- as.character(symnum(
	junc_p_vals,
	cutpoints = cutpoints,
	symbols = symbols
)[2])
nonjunc_sig <- as.character(symnum(
	nonjunc_p_vals,
	cutpoints = cutpoints,
	symbols = symbols
)[2])
pre_sig <- as.character(symnum(
	pre_p_vals,
	cutpoints = cutpoints,
	symbols = symbols
)[2])
post_sig <- as.character(symnum(
	post_p_vals,
	cutpoints = cutpoints,
	symbols = symbols
)[2])

# calculate mean of each experiment for each condition
exp_means <- data %>%
	group_by(Experiment, Dataset) %>%
	summarize(mean_CHA = mean(CHA), .groups = 'drop')

# get min and max value for plotting purposes
ymin = min(data$CHA)
ymax = max(data$CHA)

# set significance bar y-nudge value
nudge = (ymax - ymin) * 0.075

# set significance value y-nudge
sig_nudge = (ymax - ymin) * 0.03

# helper function to adjust vertical justification of significance codes
get_vjust <- function(label) {
	ifelse(grepl("\\*", label), 0.75, 0.25)
}

# helper function to hide decimal places in y-axis
dec_func <- function(x) sprintf("%.0f", x)

### create superplot
g <- ggplot() +

	# individual FOV median measurements
	geom_beeswarm(
		data = data,
		aes(x = Dataset, y = CHA, color = Experiment),
		alpha = 0.4,
		shape = 16,
		cex = 3,
		size = 1.1,
		method = 'center', # controls aesthetics of point distribution
		corral = 'random',
		preserve.data.axis=TRUE
	) +

	# mean points for each experiment
	geom_beeswarm(
		data = exp_means,
		aes(x = Dataset, y = mean_CHA, fill = Experiment),
		color = 'black',
		shape = 23,
		size = 1.75,
		cex = 2,
		method = 'center',
		preserve.data.axis=TRUE
	) +

	# model means
	geom_point(
		data = junc_data,
		aes(x = Dataset, y = Model_Fit),
		color = 'black',
		shape = 18,
		size = 1
	) +
	geom_point(
		data = nonjunc_data,
		aes(x = Dataset, y = Model_Fit),
		color = 'black',
		shape = 18,
		size = 1
	) +
	geom_point(
		data = pre_data,
		aes(x = Dataset, y = Model_Fit),
		color = 'black',
		shape = 18,
		size = 1
	) +
	geom_point(
		data = post_data,
		aes(x = Dataset, y = Model_Fit),
		color = 'black',
		shape = 18,
		size = 1
	) +

	# plot significance bars
	geom_segment(
		aes(x = 1, xend = 2, y = ymax + 4*nudge, yend = ymax + 4*nudge),
		color = 'black'
	) +
	geom_segment(
		aes(x = 1, xend = 3, y = ymax + 3*nudge, yend = ymax + 3*nudge),
		color = 'black'
	) +
	geom_segment(
		aes(x = 2, xend = 4, y = ymax + 2*nudge, yend = ymax + 2*nudge),
		color = 'black'
	) +
	geom_segment(
		aes(x = 3, xend = 4, y = ymax + nudge, yend = ymax + nudge),
		color = 'black'
	) +

	# plot significance codes
	annotate(
		"text",
		x = 1.5,
		y = ymax + 4*nudge + sig_nudge,
		label = junc_sig,
		size = 2,
		vjust = get_vjust(junc_sig)
	) +
	annotate(
		"text",
		x = 2,
		y = ymax + 3*nudge + sig_nudge,
		label = pre_sig,
		size = 2,
		vjust = get_vjust(pre_sig)
	) +
	annotate(
		"text",
		x = 3,
		y = ymax + 2*nudge + sig_nudge,
		label = post_sig,
		size = 2,
		vjust = get_vjust(post_sig)
	) +
	annotate(
		"text",
		x = 3.5,
		y = ymax + nudge + sig_nudge,
		label = nonjunc_sig,
		size = 2,
		vjust = get_vjust(nonjunc_sig)
	) +

	# SEM error bars (from emmeans)
	geom_errorbar(
		data = junc_sem_data %>% filter(Treatment == "Pre"),
		aes(x = Dataset, y = emmean, ymin = emmean-SE, ymax = emmean+SE),
		width = 0.3
	) +
	geom_errorbar(
		data = junc_sem_data %>% filter(Treatment == "Post"),
		aes(x = Dataset, y = emmean, ymin = emmean-SE, ymax = emmean+SE),
		width = 0.3
	) +
	geom_errorbar(
		data = nonjunc_sem_data %>% filter(Treatment == "Pre"),
		aes(x = Dataset, y = emmean, ymin = emmean-SE, ymax = emmean+SE),
		width = 0.3
	) +
	geom_errorbar(
		data = nonjunc_sem_data %>% filter(Treatment == "Post"),
		aes(x = Dataset, y = emmean, ymin = emmean-SE, ymax = emmean+SE),
		width = 0.3
	) +

	# axis labels
  labs(
  	x = "",
    y = bquote("Scan Area"~(µm^2))
  ) +

  # remove x label to prevent extraneous padding
	theme(axis.title.x = element_blank()) +

	theme_classic() +

	theme(
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 10),
    legend.position = "none"
  ) +

  # angle x-axis labels
  theme(axis.text.x = element_text(angle = 30, vjust = 1, hjust = 1)) +

  # adjust digits shown after decimal (e.g. accuracy = 0.01 -> two digits after decimal)
  scale_y_continuous(labels = label_number(accuracy = 1))


# create filename for plot export
savename_pdf <- paste0(safe_treatment_name, "_LMEM_all_emmeans-SEM_Scan-Area_draft", ".pdf")

ggsave(savename_pdf, plot = g, units = "in", height = 3, width = 2.5, dpi = 300)

print(g)

# capture model summaries to .txt file
junc_summ_name <- paste0(safe_treatment_name, "_LMEM_Junction_Scan-Area_summary", ".txt")
sink(junc_summ_name)
print(junc_model_summary)
print(junc_sem_data)
sink()

nonjunc_summ_name <- paste0(safe_treatment_name, "_LMEM_Non-junction_Scan-Area_summary", ".txt")
sink(nonjunc_summ_name)
print(nonjunc_model_summary)
print(nonjunc_sem_data)
sink()

pre_summ_name <- paste0(safe_treatment_name, "_LMEM_Pre_Scan-Area_summary", ".txt")
sink(pre_summ_name)
print(pre_model_summary)
print(pre_sem_data)
sink()

post_summ_name <- paste0(safe_treatment_name, "_LMEM_Post_Scan-Area_summary", ".txt")
sink(post_summ_name)
print(post_model_summary)
print(post_sem_data)
sink()

#
#

cat('\n') # terminal spacing for readability
