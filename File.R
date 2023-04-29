#Import Libraries:
library(ISLR)
library(caret)
library(class)
library(rattle)
library(pROC)

#Import Dataset:
data("GermanCredit")
str(GermanCredit[, 1:10])

#Data Preprocessing: 
sum(is.na(GermanCredit)) #Check missing values
GermanCredit[,c("Purpose.Vacation","Personal.Female.Single")] <- list(NULL) #Delete variables having same values across both classes

#Test-Train Split:
set.seed(22)
trainIndex=createDataPartition(GermanCredit$Class,p=0.7,list=FALSE)
train.feature=GermanCredit[trainIndex,-10] #Training Features
train.label=GermanCredit$Class[trainIndex] #Training Labels
test.feature=GermanCredit[-trainIndex,-10] #Test Features
test.label=GermanCredit$Class[-trainIndex] #Test Labels

#Set up Train Control:
fitcontrol=trainControl(method = "repeatedcv", number = 5, classProbs =TRUE, summaryFunction = twoClassSummary) 

#DECISION TREE:
#Training Process:
set.seed(200)
DT.rpart= train(train.feature,train.label,
                    method = "rpart",
                    metric = "ROC", #Imbalanced Data- Accuracy is not the correct metric
                    tuneLength= 12, trControl = fitcontrol)

plot(DT.rpart,col='deeppink4',pch=20,main='      ROC vs Complexity Parameter')
DT.rpart$bestTune

#Look at the details of tree:
print(DT.rpart$finalModel)

#Plotting:
fancyRpartPlot(DT.rpart$finalModel,main='Decision Tree')

#Test Error Rate:
prob1=predict(DT.rpart,test.feature, type = 'prob')
pred1=predict(DT.rpart,test.feature)
acc1=mean(pred1==test.label)
cat('Test Error Rate:', round((1-acc1),digits = 3))

#RANDOM FOREST:
#Training Process:
set.seed(370)
RF.rpart=train(train.feature, train.label,
            method="rf",
            metric="ROC", #Imbalanced Data- Accuracy is not the correct metric
            ntree = 1000,
            preProcess = c("center","scale"),
            tuneLength=12, trControl=fitcontrol)

plot(RF.rpart,col='blue4',pch=20,main='             ROC vs Mtry')
RF.rpart$bestTune

#Look at the details of RF:
print(RF.rpart$finalModel)

#Variable Importance Plot:
VarImp <- varImp(RF.rpart, scale = FALSE)
plot(VarImp,top=20,col='olivedrab',main='                               Variable Importance Plot')

#Test Error Rate:
prob2=predict(RF.rpart,test.feature, type = 'prob')
pred2=predict(RF.rpart,test.feature)
acc2=mean(pred2==test.label)
cat('Test Error Rate:', round((1-acc2),digits = 3))

#ROC CURVES:
DT.ROC <- roc(predictor=prob1$Good,response=test.label)
RF.ROC <- roc(predictor=prob2$Good,response=test.label)
plot(DT.ROC,main="ROC Curves",col='seagreen',asp=NA)
plot(RF.ROC,add=TRUE,col='indianred')
legend("bottomright", legend=c("Decision Tree", "Random Forest"), col=c("seagreen", "indianred"),lwd=3)





