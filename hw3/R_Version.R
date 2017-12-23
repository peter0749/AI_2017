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
data <- fread('~/TraData.csv', sep=',', stringsAsFactors = TRUE)

USE_FACTOR = TRUE
TUNE_SVM = FALSE
TRAIN_XGBOOST = TRUE

if (USE_FACTOR) {
  data$campaignId <- as.factor(data$campaignId)
  data$advertiserId <- as.factor(data$advertiserId)
  data$click <- as.factor(data$click)
}else{
  data$campaignId <- as.numeric(data$campaignId)
  data$advertiserId <- as.numeric(data$advertiserId)
  data$click <- as.numeric(data$click)
}

printf('We only select adx, spaceType, ip as our training feature.\n')
data <- data[,c(1,2,3,6,13)]
pos_data <- data[data$click==1, ]
neg_data <- data[data$click==0, ]
rm(data)
pos_data<-pos_data[sample(nrow(pos_data)),] ## shuffle
neg_data<-neg_data[sample(nrow(neg_data)),] ## shuffle
nfolds=10
pos_folds <- cut(seq(1,nrow(pos_data)),breaks=nfolds,labels=FALSE)
neg_folds <- cut(seq(1,nrow(neg_data)),breaks=nfolds,labels=FALSE)

data = data.frame()
folds = c()
for (i in 1:nfolds) {
  temp = rbind(pos_data[i==pos_folds], neg_data[i==neg_folds])
  temp = temp[sample(nrow(temp)), ]
  data = rbind(data, temp)
  folds = c(folds, rep(i,nrow(temp)))
}
names(data) = names(pos_data)
rm(list=c('neg_folds', 'pos_folds', 'neg_data', 'pos_data', 'temp'))
gc()

f1_score <- function(preds,labels) {
  #t = table(pred=preds, real=labels)
  recall = sum(preds==1 & labels==1) / sum(labels==1)
  precise = sum(preds==1 & labels==1) / sum(preds==1)
  f1 <- 2 * recall * precise / (recall + precise)
  acc <- sum(diag(t)) / length(labels)
  #print(t)
  #printf('acc: %.2f%%', 100.*acc)
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
  #gamma = 0.2,                   # 用於控制是否後剪枝的引數,越大越保守，一般0.1、0.2這樣子。 0.2
  max_depth = 7,                 # 構建樹的深度，越大越容易過擬合 5
  #lambda = 2.2,                  # 控制模型複雜度的權重值的L2正則化項引數，引數越大，模型越不容易過擬合。
  subsample = 0.9,               # 隨機取樣訓練樣本
  colsample_bytree = 0.9,        # 生成樹時進行的列取樣
  silent = 0,                    # 設定成 0 輸出 log 到 stderr
  eta = 0.01,                    # 學習率
  scale_pos_weight = 10.2,         # 給正樣本的權重
  nthread = 4                    # cpu 執行緒數
)

weights <- table(data$click)
weights[1] = 1
weights[2] = 10.2

if(TUNE_SVM) {
  gc()
  require('foreach')
  require('doParallel')
  registerDoParallel(cores = 4)
  ### PARAMETER LIST ###
  cost <- c(0.1,1,10,100,1000)
  gamma <- c(0.1,0.5,1,2,4)
  parms <- expand.grid(cost = cost, gamma = gamma)
  ### LOOP THROUGH PARAMETER VALUES ###
  params_result <- foreach(i = 1:nrow(parms), .combine = rbind) %do% {
    c <- parms[i, ]$cost
    g <- parms[i, ]$gamma
    ### K-FOLD VALIDATION ###
    out <- foreach(j = 1:nfolds, .combine = rbind, .inorder = FALSE) %dopar% {
      deve <- data[folds != j, ]
      test <- data[folds == j, ]
      ub <- ubBalance(deve[,-5], as.factor(unlist(deve[,5])), type='ubUnder', positive=1)
      sparse_X = sparse.model.matrix(~ ., data = ub$X)
      sparse_X.svd = svd(sparse_X)
      mdl <- e1071::svm(sparse.model.matrix(~ ., data = ub$X), as.factor(ub$Y), type = "C-classification",
                        kernel = "radial", cost = c, gamma = g #,
                        #class.weights = weights
      )
      pred <- predict(mdl, sparse.model.matrix(click ~ .-1, data = test))
      data.frame(y_true = test$click, y_pred = pred)
    }
    printf('Tuning progress: %d / %d\n', i, nrow(parms))
    ### CALCULATE SVM PERFORMANCE ###
    f1_score_val <- f1_score(out$y_pred, out$y_true)
    data.frame(parms[i, ], f1 = f1_score_val)
  }
  gc()
}

if (TRAIN_XGBOOST){
  f1_records = c()
  for (i in 1:nfolds) {
    validIdx = which(folds==i, arr.ind=TRUE)
    gc()
    # ub = ubBalance(data[-validIdx,-5], as.factor(unlist(data[-validIdx,5])), type='ubUnder', positive=1)
    # print(count((as.numeric(ub$Y)-1)))
    ub = list(X=data[-validIdx,-5], Y=as.factor(unlist(data[-validIdx,5])))
    bst <- xgboost(params=xgbTreeParams, data=sparse.model.matrix(~ ., data = ub$X), label=as.integer(ub$Y)-1, nrounds=20,  verbose = TRUE)
    # bst <- rpart(click ~ . , data = cbind(ub$X, click=ub$Y), method='class')
    # bst <- svm(sparse.model.matrix(~ ., data = ub$X), ub$Y, type='one-classification')
    # bst <- svm(click ~ . , data = cbind(ub$X, click=ub$Y), type='one-classification')
    # bst <- randomForest(click ~ . , data = cbind(ub$X, click=ub$Y), method='class')
    # bst <- glm(click ~ . , data = cbind(ub$X, click=ub$Y), family=binomial)
    gc()
    pred <- predict(bst, sparse.model.matrix(~ ., data = data[validIdx,-5]))
    # pred <- predict(bst, data[validIdx,-5])
    f1 <- f1_score(as.integer(pred>0.5), as.numeric(unlist(data[validIdx,5]))-1)
    printf('f1(valid): %.4f\n', f1)
    f1_records = append(f1_records, f1)
  }
  print(summary(f1_records))
}
