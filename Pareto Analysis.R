rm(list=ls())
library(devtools)
library(rgl)
library(OpenMORDM)
library(MASS)
library(scales)
library(rgdal)
library(stringr)
library(plotly)
# library(oceanmap)

master.folder <- "C:/Users/Josh Soper/Documents/Master's Thesis/Modeling/Josh/JMET-TRIAL"
setwd(master.folder)
model <- "2018_trial1"
model.folder <- file.path(master.folder, model, 'init')
variables <- c("EXH2O", 'Intake_Elev', 'WSC_THOMAS', "WSC_SOUTH", "WSC_NORTH")
objectives <- c('BN_TEMP', "BN_COND", "CI_TEMP", 'CI_Cond')
functions <- list.files("C:/Users/Josh Soper/Documents/Master's Thesis/Modeling/Josh/R/Functions", full.names = T)
lapply(functions, source)
parcoordJS <- function (x, n, col = 1, lty = 1, var.label = FALSE, ...) 
{
  rx <- apply(x, 2L, range, na.rm = TRUE)
  x <- apply(x, 2L, function(x) (x - min(x, na.rm = TRUE))/(max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
  matplot(1L:ncol(x), t(x), type = "l", col = col, lty = lty, 
          xlab = "", ylab = "", axes = FALSE, ...)
  axis(1, at = 1L:ncol(x), labels = colnames(x), tick = 5)
  for (i in 1L:ncol(x)) {
    lines(c(i, i), c(0, 1), col = alpha('gray50', 0.5))
    if (var.label) 
      text(c(i, i), c(0, 1), labels = format(rx[, i], digits = 3), 
           xpd = NA, offset = 0.3, pos = c(1, 3), cex = n)
  }
  invisible()
}
# RESULTS -----------------------------------------------------------------
return.results <- function(model, headers) {
  vars <- read.table(file.path("results", paste("VAR_NSGAII_cequal", model, ".set", sep = '')))
  objs <- read.table(file.path("results", paste("OBJ_NSGAII_cequal", model, ".set", sep = '')))
  results <- cbind(vars,objs)
  names(results) <- headers
  return(results)
}

append.cosgrove.cond <- function(results, model, headers) {
  all.runs <- read.csv(file.path("results", paste("cequal", model, "_all_runs.txt", sep = '')), header = F)[-10]
  names(all.runs) <- headers
  all.runs <- merge(results, all.runs, by = names(results), all = F, no.dups = F)
  all.runs <- unique(all.runs)
  return(all.runs)
}

headers <- append(variables, append(objectives, "CI_COND"))
results <- return.results(model, append(variables,objectives))
head(results)
# results <- append.cosgrove.cond(results, model, headers)

# PLOTS AND VISUALIZATION -------------------------------------------------
# order by temperature, then add bounds
pareto <- results[order(results$CI_TEMP),]

pareto[nrow(pareto)+1,] <- c(0.25, 104.3, 0.5, 0.5, 0.5, 0.5, 5, 0.5, 5)
pareto[nrow(pareto)+1,] <- c(0.65, 110.6, 1, 1, 1, 2.5, 15, 2.5, 15)

col <- colorRampPalette(c("blue", 'cornflowerblue', "pink", 'red'))(nrow(pareto)-2)
col <- append(col,rep("#000000", 2))
trans = 0.5
# bar.ticks <- round(pareto$CI_TEMP[1]:pareto$CI_COND[nrow(pareto)], 2)

pareto.obj.plot <- function(pareto, write.plot = T) {
  if (write.plot == T) {png(paste("analysis/", model, "obj", '.png', sep =''), width = 6, height = 5, units  ="in", res = 300)}
  par(oma = c(0,1,0,1), mgp = c(1,2,0.5))
  parcoordJS(pareto[,6:9], n = 1, col = alpha(col, trans), var.label = T, lwd = 3, xaxt = 'n')
  title(ylab = 'RMSE (°C or µS/cm)', cex.main = 1.5, cex.lab = 1.25)
  if (write.plot == T) {dev.off()}
}

pareto.var.plot <- function(pareto, write.plot = T) {
  if (write.plot == T) {png(paste("analysis/", model, '.png', sep =''), width = 11, height = 5, units  ="in", res = 300)}
  par(oma = c(0,1,0,1), mgp = c(1,2,0.5))
  parcoordJS(pareto[,1:4], n = 1, col = alpha(col, trans), var.label = T, lwd = 3)
  title(ylab = 'Model Parameter Value', cex.main = 1.5, cex.lab = 1.25)
  if (write.plot == T) {dev.off()}
}
pareto.full.plot <- function(pareto, write.plot = T) {
  if (write.plot == T) {png(paste("analysis/", model, '.png', sep =''), width = 11, height = 5, units  ="in", res = 300)}
  par(oma = c(0,1,0,1), mgp = c(1,2,0.5))
  parcoordJS(pareto, n = 1, col = alpha(col, trans), var.label = T, lwd = 3)
  title(ylab = 'Model Parameter Value', cex.main = 1.5, cex.lab = 1.25)
  if (write.plot == T) {dev.off()}
}

# p <- plot_ly(pareto, x = ~BN_TEMP, y = ~BN_COND, z = ~CI_TEMP) %>%
#   add_markers()
# p

pareto.full.plot(pareto, write.plot = F)


results[results$BN_TEMP<1.3 & results$CI_TEMP < 1.3,]

