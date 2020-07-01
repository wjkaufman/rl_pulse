library(ggplot2)

setwd("~/projects/rl_pulse/data/2020-07-01-eval/")

df = read.csv("candidate4.csv")
#df$type[df$type=="candidate"] = "Candidate"
df$reward = -log10(1-df$fidelity + 1e-100)
# df$time = ifelse(df$type=="Candidate", 3*df$delay + 12*df$pulseWidth, 6*df$delay + 4*df$pulseWidth)
df[,c("delay", "pulseWidth")] = signif(df[,c("delay", "pulseWidth")], digits = 2)

g = ggplot(df, aes(x=reward, group=type, fill=type)) +
  geom_histogram(bins=50, alpha = 0.6, position = 'identity') +
  facet_grid(rows=vars(delay), cols=vars(pulseWidth), labeller = label_both) +
  theme_minimal() +
  labs(x="Reward", y="Count", fill="Sequence")
print(g)
ggsave("reward_hist.pdf", width = 8, height=6)

g = ggplot(df, aes(x=delay, y=time, size=pulseWidth, group=type, color=type)) +
  geom_point()
print(g)

g = ggplot(df, aes(x=time)) + geom_histogram()
print(g)

g = ggplot(df, aes(x=rep, group=type, fill=type)) + geom_histogram(position = "identity", alpha=0.7)
print(g)


