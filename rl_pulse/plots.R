library(tidyr)
library(dplyr)
library(ggplot2)

setwd('~/projects/rl_pulse/data/2020-06-23/')

# create plots for each directory that matches regex pattern
for (datFolder in list.files(pattern = "00*")) {
  # navigate to directory
  setwd(datFolder)
  print(paste("In directory:", datFolder))
  suffix = substr(datFolder, 0, 3)
  # make plots
  paramDiff = read.csv("paramDiff.csv")
  paramDiff$layerNum = factor(paramDiff$layerNum, ordered = T)
  for (networkType in c("actor", "critic")) {
    g = ggplot(paramDiff %>% filter(type==networkType),
               aes(x=generation, y=diff, group=layerNum, color=layerNum)) +
      geom_line() + 
      labs(title=paste(networkType, "parameter difference"),
           x='Generation',
           y='Parameter difference',
           color="Layer Number")+
      scale_y_log10()
      theme_bw()
    # print(g)
    ggsave(paste0("../graphics/", networkType, "_paramDiff-", suffix, ".png"),
           scale=1.5, width=5, height=3, units="in")
  }
  
  popFitnesses = read.csv("popFitnesses.csv")
  popFitnesses$individual = as.factor(popFitnesses$individual)
  g = ggplot(popFitnesses,
             aes(x=generation, y=fitness, group=individual, size=fitnessInd,
                 color=synced)) +
    geom_point(alpha=0.8) + 
    labs(title="Population fitnesses",
         x='Generation',
         y='Fitness',
         size="Max reward index",
         color="Synced")+
    theme_bw()
  # print(g)
  ggsave(paste0("../graphics/popFitnesses-", suffix, ".png"), scale=1.5, width=5, height=3, units="in")
  
  testFitnesses = read.csv("testMat.csv")
  g = ggplot(testFitnesses,
             aes(x=generation, y=fitness, color=type, size=fitnessInd)) +
    geom_point() + 
    labs(title="Test fitnesses",
         x='Generation',
         y='Fitness',
         color='Type',
         size="Max reward index")+
    # scale_y_log10() + 
    theme_bw()
  # print(g)
  ggsave(paste0("../graphics/testFitnesses-", suffix, ".png"), scale=1.5, width=5, height=3, units="in")
  
  # pack out to parent directory
  setwd("..")
}
