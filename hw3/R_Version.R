rm(list=ls(all=TRUE))
require(pbapply)
require(foreach)
require(doParallel)
no_cores <- detectCores() - 1  
registerDoParallel(cores=no_cores) 
cl <- makeCluster(no_cores)
require(data.table)
require(rpart)
require(rpart.plot)
require(caret)

printf <- function(...) cat(sprintf(...)) # printf function
data <- fread('./AI_2017/hw3/TraData.csv', sep=',', stringsAsFactors = TRUE)
data$campaignId <- as.factor(data$campaignId)
data$advertiserId <- as.factor(data$advertiserId)
data$click <- as.factor(data$click)
# data <- data[sample(nrow(data), 1000), ]
# 刪除下面幾行註解，來觀察 ANOVA 結果
#subset <- data[sample(nrow(data), 3000), ]

#printf("click sum: %d\n",sum(subset$click==1))
#printf("click ratio: %.2f\n",sum(subset$click==1) / nrow(subset))
#anova <- aov(data=subset, click ~ .)
#summary(anova)
#subset=NULL

data = data[,c(1,2,3,6,13)] # adx, spaceType, spaceId, ip, click
data<-data[sample(nrow(data)),] ## shuffle
folds <- cut(seq(1,nrow(data)),breaks=10,labels=FALSE)

train_tree <- function(i, data, folds) {
  result <- try(
    {
      validIdx = which(folds==i, arr.ind=TRUE)
      if(sum(data[-validIdx, ]==1)==0 || sum(data[validIdx, ]==1)==0){
        return(0)
      }
      propose_clf_tree <- rpart::rpart(click ~ . , data=data[-validIdx, ], method='class')
      pred <- as.data.frame(predict(propose_clf_tree,data[validIdx,-5]))
      pred_pos = pred[,2] > pred[,1]
      valid.table <- table(pred=pred_pos, true=(data[validIdx,5]==1))
      recall = valid.table[2,2]/sum(valid.table[,2])
      precision = valid.table[2,2]/sum(valid.table[2,])
      f1 = 2*precision*recall/(precision+recall)
      return(f1)
    }, silent=TRUE
  )
  if (class(result) == "try-error") {
    return(0)
  } 
  return(result)
}

results <- pblapply(1:10, train_tree, data=data, folds=folds, cl=cl)
stopCluster(cl) 
results
