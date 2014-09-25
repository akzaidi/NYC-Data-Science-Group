source("../../../R/AZ Functions.R")
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)
library(caret)
library(pROC)
library(ROCR)

## Lending club default prediction

loan.stats <- function() {
  
  dir <- "Lending Club Data/"
  files <- list.files(dir, full.names = T)
  l.stats <- files[str_detect(files, ignore.case("loanstats"))]
  
  list.stats <- lapply(l.stats, 
                       function(x) 
                         read.csv(x, skip = 1, header = T, 
                                  stringsAsFactors = F))
  
  l.stats <- gsub(dir, "", l.stats)
  names(list.stats) <- l.stats
  
  return(ldply(list.stats))
  
}

loan.stats.df <- loan.stats()

edit.lstats <- function(lstats = loan.stats.df) {
  
  lstats$loan_status <- ifelse(str_detect(lstats$loan_status, "Status"), 
                               substr(lstats$loan_status, 
                                      str_locate(lstats$loan_status, "Status:")[[2]] + 1,
                                      nchar(lstats$loan_status)),
                               lstats$loan_status)
  
  return(lstats)
}

ggplot(ddply(subset(loan.stats.df, !is.na(loan_status)), .(grade, loan_status), summarise, NLoans = nrow(piece), Total.Amount = sum(loan_amnt, na.rm = T), .progress = 'text'))


run.model <- function(lstats = loan.stats.df) {
  
  cat.predictors <- c("grade", "home_ownership", "emp_length", "purpose", "is_inc_v")
  cts.predictors <- c("annual_inc", "dti", "loan_amnt", "int_rate")
  
  lstats$loan_def <- ifelse(lstats$loan_status %in% c("Charged Off",
                                                       "Default"), 1, 0)
  lstats$loan_def <- ifelse(str_detect(lstats$loan_status, "Late"),
                            1, lstats$loan_def)
  
  lstats$int_rate <- as.numeric(gsub("%", "", lstats$int_rate))
  
  return(lstats[, c(cat.predictors, cts.predictors, "loan_def", "loan_status")])
  
}

split.data <- function(data, train.size = 0.5) {
  
  train.rows <- sample(1:nrow(data), size = floor(nrow(data)/2), replace = F)
  train.data <- data[train.rows, ]
  test.data  <- data[!(1:nrow(data) %in% train.rows), ]
  
  list.data <- list("Train" = train.data, "Test" = test.data)
  
  return(list.data)
  
}


train.tree <- function(tree.data) {
  
  fol <- formula(loan_def ~ grade + home_ownership + is_inc_v +
                   emp_length + purpose + annual_inc +
                   dti + loan_amnt)
  mod <- rpart(fol, method = 'class', data = tree.data,
               control = rpart.control(minbucket = 1, minsplit = 1, 
                                       cp = -2, xval = 0, 
                                       maxdepth = 3))
  
  return(mod)
}


train.glm <- function(tree.data) {
  
  fol <- formula(loan_def ~ grade + home_ownership + is_inc_v +
                   emp_length + purpose + annual_inc +
                   dti + loan_amnt)
  mod <- glm(fol, family = 'binomial', data = tree.data)
  
  return(mod)
}

predict.tree <- function(train.model = train.tree(loans.split[["Train"]]),
                         test.data = loans.split[["Test"]]) {
  
  train.model <- train.model
  test.predic <- predict(train.model, newdata = test.data, type = 'class',
                         control = rpart.control(minbucket = 1, minsplit = 1, 
                                                 cp = -2, xval = 0, 
                                                 maxdepth = 3))
  
  return(test.predic)
  
}


train.forest <- function(for.data = loans.split[["Train"]]) {
  
  fol <- formula(factor(loan_def) ~ grade + home_ownership +
                   emp_length + purpose + is_inc_v + annual_inc +
                   dti + loan_amnt)
  forest <- randomForest(fol, data = for.data, 
                         type = 'classification', importance = T)
  
  return(forest)
  
}

results <- function(test = predict.tree(),
                    test.data = loans.split$Test,
                    response = "loan_def") {
  
  cross.tab.tree <- table(test, test.data[, response])
  
  correct <- sum(diag(cross.tab.tree))/nrow(test.data)
  
  return(correct)
}

train.svm <- function(for.data = loans.split[["Train"]]) {
  
  fol <- formula(factor(loan_def) ~ grade + home_ownership + int_rate +
                   emp_length + purpose + is_inc_v + annual_inc +
                   dti + loan_amnt)
  svm.model <- svm(fol, data = for.data)
  
  return(svm.model)
  
}


train.boost <- function(train.data = loans.split$Train) {
  
  fol <- formula(factor(loan_def) ~ factor(grade) + factor(grade) + factor(grade) +
                   factor(grade) + factor(purpose) + annual_inc +
                   dti + loan_amnt,
                 control = rpart.control(minsplit = 0))
  
  loans.adaboost <- gbm(fol, data = train.data, 
                        distribution = 'adaboost')
  
  return(loans.adaboost)
  
}


data.edit <- function(loans.data = lstats.x) {
  
  loans.data$home_own <- ifelse(loans.data$home_ownership %in% c("OWN", "MORTGAGE"),
                                1, 0)
  
  loans.data$employed <- ifelse(loans.data$emp_length %in% c("<1 year", "n/a"),
                                0, 1)
  loans.data$small.biz <- ifelse(loans.data$purpose == "small_business", 1, 0)
  
  loans.data$verify <- ifelse(loans.data$is_inc_v %in% c("Verified", "Source Verified"),
                              1, 0)
  loans.data$high.grade <- ifelse(loans.data$grade %in% c("A", "B"), 
                                  1, 0)
  
  return(loans.data[, c("high.grade", "verify", "small.biz", "employed", 
                        "home_own", "int_rate", "loan_amnt", "dti", "annual_inc",
                        "loan_def")])
  
}

run.forest <- function(lstats = data.edit()) {
  
  splits <- split.data(lstats)
  rf <- randomForest(factor(loan_def) ~ ., 
                     data = splits$Train, importance = T, na.action = na.omit)
  
  return(rf)
}

forest.results <- function(lstats = data.edit()) {
  
  
  splits <- split.data(lstats)
  model <- run.forest()
  predictions <- predict(run.forest(), splits$Test, type = 'class')
  
  table.preds <- table(predictions, splits$Test[, 'loan_def'])
  
  return(table.preds)
  
}

pred.forest <- predict(run.forest(), splits.loans.bin$Test, type = 'class')
performance(prediction(as.numeric(as.character(pred.forest)), splits.loans.bin$Test$loan_def), 'tpr', 'fpr')
prob.forest.pred <- predict(run.forest(), splits.loans.bin$Test, type = 'prob')

plot(performance(prediction(prob.forest.pred[, 2], splits.loans.bin$Test$loan_def), "tpr", "fpr"), main = "ROC for Random Forest Default Prediction", col = 2, lwd = 2)


y <- (performance(prediction(prob.forest.pred[, 2], splits.loans.bin$Test$loan_def), "tpr", "fpr"))
ydf <- (data.frame("TPR" =  y@x.values, "FPR" = y@y.values))

## for svms  use predict(svm.fit, newdata, probability = T)
## and extract using attr(prob.svm.pred, "prob")