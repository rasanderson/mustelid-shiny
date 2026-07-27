# Check loss accuracy plots

library(tidyr)
library(ggplot2)

tmp <- read.csv("log/Crop/Plain/Plain_Crop_Res18_plain_071824/version_0/loss_accuracy.csv")

tmp2 <- pivot_longer(tmp, 
  cols = -epoch,
  names_to = c("type", ".value"),
  names_sep = "_"
)

library(ggplot2)

p_acc <- ggplot(tmp2, aes(x = epoch, y = acc, colour = type)) +
  geom_line(linewidth = 1) +
  geom_point() +
  labs(
    x = "Epoch",
    y = "Accuracy",
    colour = "Dataset"
  ) +
  theme_minimal()

p_loss <- ggplot(tmp2, aes(x = epoch, y = loss, colour = type)) +
  geom_line(linewidth = 1) +
  geom_point() +
  labs(
    x = "Epoch",
    y = "Loss",
    colour = "Dataset"
  ) +
  theme_minimal()

p_acc
p_loss
