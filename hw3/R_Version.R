rm(list=ls(all=TRUE))
require(Matrix)
require(data.table)
require(rpart)
require(rpart.plot)
require(randomForest)
require(caret)
require(xgboost)
require(e1071)
require(unbalanced)
require(plyr)

printf <- function(...) cat(sprintf(...)) # printf function
data <- fread('./AI_2017/hw3/TraData.csv', sep=',', stringsAsFactors = TRUE)
data$campaignId <- as.factor(data$campaignId)
data$advertiserId <- as.factor(data$advertiserId)
data$click <- as.factor(data$click)
printf('We only select adx, spaceType, ip as our training feature.\n')
data <- data[,c(1,2,3,6,13)]
data<-data[sample(nrow(data)),] ## shuffle
nfolds=10
folds <- cut(seq(1,nrow(data)),breaks=nfolds,labels=FALSE)

sumwpos <- sum(data$click==1)
sumwneg <- nrow(data)-sumwpos
pos_weight = sumwneg / sumwpos

f1_score <- function(preds,labels) {
  recall = sum(preds==1 & labels==1) / sum(labels)
  precise = sum(preds==1 & labels==1) / sum(preds)
  f1 <- 2 * recall * precise / (recall + precise)
  return(f1)
}

f1_metric <- function(prob, dtrain) {
  preds = as.integer(prob>0.5)
  labels <- getinfo(dtrain, "label")
  f1 = f1_score(preds,labels)
  return(list(metric = "f1", value = f1))
}

xgbTreeParams <- list(
  objective='binary:logistic',
  eval_metric=f1_metric,
  #scale_pos_weight = pos_weight,
  nthread=4
)
f1_records = c()
for (i in 1:nfolds) {
  validIdx = which(folds==i, arr.ind=TRUE)
  gc()
  ub = ubBalance(data[-validIdx,-5], as.factor(unlist(data[-validIdx,5])), type='ubOver')
  # print(count((as.numeric(ub$Y)-1)))
  # ub = list(X=data[-validIdx,-5], Y=as.factor(unlist(data[-validIdx,5])))
  bst <- xgboost(params=xgbTreeParams, data=sparse.model.matrix(~ ., data = ub$X), label=as.integer(ub$Y)-1, nrounds=120,  verbose = TRUE)
  # bst <- rpart(click ~ . , data = cbind(ub$X, click=ub$Y), method='class')
  # bst <- svm(sparse.model.matrix(~ ., data = ub$X), ub$Y, type='one-classification')
  # bst <- randomForest(click ~ . , data = cbind(ub$X, click=ub$Y), method='class')
  gc()
  pred <- predict(bst, sparse.model.matrix(~ ., data = data[validIdx,-5]))
  f1 <- f1_score(as.integer(pred>0.5), as.numeric(unlist(data[validIdx,5]))-1)
  printf('f1(valid): %.4f\n', f1)
  f1_records = append(f1_records, f1)
}

print(summary(f1_records))

