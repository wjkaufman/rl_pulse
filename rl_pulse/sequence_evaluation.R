library(ggplot2)
library(dplyr)

setwd("~/projects/rl_pulse/data/2020-07-01-eval/")

df.whh = read.csv("WHH.csv")
df = read.csv("candidate4.csv")
df = bind_rows(df, df.whh)
#df$type[df$type=="candidate"] = "Candidate"
#df$reward = -log10(1-df$fidelity + 1e-100)
# df$time = ifelse(df$type=="Candidate", 3*df$delay + 12*df$pulse_width, 6*df$delay + 4*df$pulse_width)
df[,c("delay", "pulse_width")] = signif(df[,c("delay", "pulse_width")], digits = 2)

g = ggplot(df, aes(x=reward, group=type, fill=type)) +
  geom_histogram(bins=50, alpha = 0.6, position = 'identity') +
  facet_grid(rows=vars(delay), cols=vars(pulse_width), labeller = label_both) +
  theme_minimal() +
  labs(x="Reward", y="Count", fill="Sequence")
print(g)
ggsave("reward_hist.pdf", width = 8, height=6)

g = ggplot(df, aes(x=delay, y=time, size=pulse_width, group=type, color=type)) +
  geom_point()
print(g)

g = ggplot(df, aes(x=time)) + geom_histogram()
print(g)

g = ggplot(df, aes(x=rep, group=type, fill=type)) + geom_histogram(position = "identity", alpha=0.7)
print(g)



#### WHH ####

temp = df.whh %>% group_by(delay, pulse_width, type, time) %>% 
  summarize(fidelity.mean = mean(fidelity), fidelity.sd = sd(fidelity))

g = ggplot(temp, aes(x = time, y = fidelity.mean)) +
  facet_grid(rows = vars(delay), cols = vars(pulse_width)) +
  geom_point() + geom_line() + geom_errorbar(aes(ymin = fidelity.mean - fidelity.sd,
                                   ymax = fidelity.mean + fidelity.sd)) +
  scale_y_continuous(trans = "exp")
g
