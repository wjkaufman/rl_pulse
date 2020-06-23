library(tidyr)
library(dplyr)
library(ggplot2)

# create plots for each directory that matches regex pattern
for (datFolder in list.files(pattern = "00*")) {
  # navigate to directory
  setwd(datFolder)
  print(paste("In directory:", datFolder))
  # make plots
  paramDiff = read.csv("paramDiff.csv")
  for (networkType in c("actor", "critic")) {
    g = ggplot(paramDiff %>% filter(type==networkType),
               aes(x=generation, y=diff, group=layerNum, color=layerNum)) +
      geom_line(size=1) + 
      labs(title=paste(networkType, "parameter difference"),
           x='Generation',
           y='Parameter difference',
           color="Layer Number")+
      # scale_y_log10() + 
      theme_bw()
    # print(g)
    ggsave(paste0(networkType, "_paramDiff.png"), scale=1.5, width=5, height=3, units="in")
  }
  
  popFitnesses = read.csv("popFitnesses.csv")
  popFitnesses$individual = as.factor(popFitnesses$individual)
  g = ggplot(popFitnesses,
             aes(x=generation, y=fitness, group=individual, shape=individual, size=fitnessInd,
                 color=synced)) +
    geom_point(alpha=0.8) + 
    labs(title="Population fitnesses",
         x='Generation',
         y='Fitness',
         size="Max reward index",
         color="Individual")+
    # scale_y_log10() + 
    theme_bw()
  # print(g)
  ggsave("popFitnesses.png", scale=1.5, width=5, height=3, units="in")
  
  testFitnesses = read.csv("testMat.csv")
  g = ggplot(testFitnesses,
             aes(x=generation, y=fitness, color=type, size=fitnessInd)) +
    geom_point() + 
    labs(title="Test fitnesses",
         x='Generation',
         y='Fitness',
         size="Max reward index")+
    # scale_y_log10() + 
    theme_bw()
  # print(g)
  ggsave("testFitnesses.png", scale=1.5, width=5, height=3, units="in")
  
  # pack out to parent directory
  setwd("..")
}
