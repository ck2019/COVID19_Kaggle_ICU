#Description - Script created to determine are there features in the dataset
#that can be used to predict if someone has SARS-COV-2 and needs an ICU bed and
# needs a hospital bed
#Name - Ciara Kerrigan
#Date - 03/06/2020

options(warn=-1)

library(caTools)
library(caret)
library(doParallel)
library(reshape2)
library(dplyr)
library(pROC)
library(DMwR)
library(ggplot2)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#list.files(path = "/kaggle/input/uncover/UNCOVER/einstein/")


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

set.seed(100)
rm(list=ls())


#read in data file
COV19=read.csv("B:/01. FIM/39. Innovation/Ciaras_AlteryxApps/COV19/input/uncover/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv",header=T,sep=",")
#COV19=read.csv("/kaggle/input/uncover/UNCOVER/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv",header=T,sep=",")

#convert to matrix first to force categories to integers and then back to data frame
data2=data.frame(round(data.matrix(COV19),5))

#Remove columns which are all NA
ColumnsRemove1=data2 %>% select_if(~ all(is.na(.))) %>% names()
data2 <- data2[, unlist(lapply(data2, function(x) !all(is.na(x))))]
Percent_100=NROW(ColumnsRemove1)

print("Columns that have 100% nulls and will be removed are")
print(ColumnsRemove1)

#Drop columns with 90% nulls
NApercent <- 0.90
MaxNAs = round(sum(NROW(data2)) * NApercent,0)

#find columns with 90% nulls
ColumnsRemove2=data2 %>% select_if(~ sum(is.na(.))>MaxNAs) %>% names()
data2 <- data2[, unlist(lapply(data2, function(x) sum(is.na(x))<MaxNAs))]
Percent_90=NROW(ColumnsRemove2)

print("Columns that have 90% nulls and will be removed are")
print(ColumnsRemove2)

#Create Barchart
counts=cbind(Percent_90,Percent_100)

#plot barchart showing number of features with a high % of NAs
barplot(counts, main="Number of features with high % of NAs",
        xlab="Percentage of NAs",col=c("darkgreen"), ylab="Number of features")

#set the rest of the NAs to 0
ColsNAs=data2 %>% select_if(~ any(is.na(.))) %>% names()
data2[is.na(data2)] <- 0

#no NAs so worked fine
sum(is.na(data2))



#Feature engineering
#Create a new feature to use as target feature as not enough patients tested postive for COVID and were admitted to ICU in the dataset.
data2$admittedToHospital=1
for (i in 1:NROW(data2)){
  #if (data2$patient_addmited_to_regular_ward_1_yes_0_no[i]==2 & data2$sars_cov_2_exam_result[i]==2){data2$admittedToHospital[i]=2}
  if (data2$patient_addmited_to_semi_intensive_unit_1_yes_0_no[i]==2 & data2$sars_cov_2_exam_result[i]==2){data2$admittedToHospital[i]=2}
  if (data2$patient_addmited_to_intensive_care_unit_1_yes_0_no[i]==2 & data2$sars_cov_2_exam_result[i]==2){data2$admittedToHospital[i]=2}
}

#######Test are there any issues with data to model
PreProcessCheck1=preProcess(data2,method= c("center", "scale","knnImpute"))

#######Now removes varaiables with zero variance as cannot be used for predicion
data1a=subset(data2, select = -c(urine_nitrite))
PreProcessCheck1=preProcess(data1a,method= c("center", "scale","knnImpute"))

###########################################
# check for highly correlated variables
###########################################
Cor1P=round(cor(data.matrix(data1a), method = "pearson", use="complete.obs"),1)

hc1=0.9  
Details=paste("Definition of high correlation:",hc1)
print(Details)

hc = findCorrelation(Cor1P, cutoff=hc1)
hc = sort(hc)
data1 = data1a[,-c(hc)]

p1=6/10  
Details=paste("Ratio of training to overall:",p1)
print(Details)

#breakup into test and train again now that issues are fixed
inTrain=createDataPartition(y=data1$admittedToHospital,times=1,p=p1,list=FALSE)
training=data1[inTrain,]
testing=data1[-inTrain,]

Details=paste("Number of rows in training is:",NROW(training))
print(Details)

Details=paste("Number of rows in testing is:",NROW(testing))
print(Details)

#determine proportions
out1=prop.table(table(training$admittedToHospital))
out2=prop.table(table(testing$admittedToHospital))

Details=paste("Train Imbalance",out1)
print(Details)
Details=paste("Test Imbalance",out2)
print(Details)

NumTesting1=sum(testing$admittedToHospital==2)
NumTraining1=sum(training$admittedToHospital==2)


##################################################################
## Remove fields used to create target
##################################################################

TrainNoPred=subset(training, select = -c(patient_addmited_to_regular_ward_1_yes_0_no,patient_addmited_to_semi_intensive_unit_1_yes_0_no,patient_addmited_to_intensive_care_unit_1_yes_0_no,admittedToHospital,sars_cov_2_exam_result,patient_id))
TestNoPred=subset(testing, select = -c(patient_addmited_to_regular_ward_1_yes_0_no,patient_addmited_to_semi_intensive_unit_1_yes_0_no,patient_addmited_to_intensive_care_unit_1_yes_0_no,admittedToHospital,sars_cov_2_exam_result,patient_id))

######################Test are there any issues with data to model
PreProcessCheck2=preProcess(TrainNoPred,method= c("center", "scale","knnImpute"))
PreProcessCheck2



#Set cross validation to 5
cvnumber=5
Details=paste("Cross validation used is",cvnumber)
print(Details)


#set the outcome so can be used in the training functions
outcome=as.factor(training$admittedToHospital)
outcomeReg=as.factor(training$admittedToHospital)
a3=factor(outcome, labels = make.names(levels(outcome)))
outcome=a3
outcome2=outcome


#Now get actuals
actuals1=as.factor(testing$admittedToHospital)
a3=factor(actuals1, labels = make.names(levels(actuals1)))
actuals=a3

actualsTrain=as.factor(training$admittedToHospital)
a3=factor(actualsTrain, labels = make.names(levels(actuals)))
actualsTrain=a3


########################################################
##Creating a customised SMOTE function as better results seen with this

smotest3 <- list(name = "SMOTE with more neighbors!",
                 func = function (x, y) {
                   library(DMwR)
                   dat <- if (is.data.frame(x)) x else as.data.frame(x)
                   dat$.y <- y
                   dat <- SMOTE(.y ~ ., data = dat, k = 5,perc.over = 1200, perc.under=300)
                   list(x = dat[, !grepl(".y", colnames(dat), fixed = TRUE)], 
                        y = dat$.y)
                 },
                 first = TRUE)


#setup train control parameters
ctrl3 <- trainControl(method = "repeatedcv", 
                      number = cvnumber, 
                      repeats = 5,
                      sampling = smotest3
) 



Details=paste("Start of regression GBM SMOTE",Sys.time())
print(Details)
TrainGBM3=train(x=TrainNoPred,y=outcome,method ="gbm",trControl=ctrl3,preProcess = c("center", "scale","knnImpute"))           
Details=paste("End of regression GBM SMOTE",Sys.time())
print(Details)

#Print out training result
details=paste("Training result",TrainGBM3$results)
print(details)

#Run prediction on test data
predictedGBM3=data.frame(predict(TrainGBM3,newdata=data.matrix(TestNoPred),type="raw"))
predictionsGBM3=factor(predictedGBM3$predict.TrainGBM3..newdata...data.matrix.TestNoPred...type....raw..)

#confusion matrix
cv=confusionMatrix(predictionsGBM3, actuals, mode="everything")

print("Confusion Matrix Test prediction results for GBM SMote")
print(cv$overall)
print("More Confusion Matrix Test prediction results for GBM SMote") 
print(cv$byClass)

#determine other performance metrics such as R^2
Performance1=data.frame(cbind(actuals,predictionsGBM3))
Performance2=rename(Performance1,obs=actuals, pred=predictionsGBM3)
Performance3=caret::defaultSummary(data=Performance2,  lev=c("1","2"),model = "gbm")
print("Other performance metrics") 
print(Performance3)

#Detetemine influence of certain variables
x1=TrainGBM3$finalModel
x2=data.frame(summary(x1))

#Look at top 10 most important relative predictors                               
dt1=data.frame(x2[1:10,])
x=factor(dt1$var, levels=dt1$var)
y=dt1$rel.inf

#plot top 10
ggplot() + 
  geom_bar(data = dt1, aes(x=x, y=y), stat="identity",fill="red") +
  coord_flip() + ggtitle("Relative importance of top 10 predictors") +
  labs(y= "Relative Importance", x = "Predictor")


names1=TrainGBM3$finalModel$var.names
#find the index of variables want to run biplots for
for (i in 1:NROW(names1))
{ if (names1[i]=="patient_age_quantile")
{ age1=i}
  if (names1[i]==x[1])
  { var1=i}
  if (names1[i]==x[2])
  { var2=i}
  if (names1[i]==x[3])
  { var3=i}
}


#Plot Marginal effect of age and eosinophils
plot(TrainGBM3$finalModel,i.var = cbind(var1,age1),
     n.trees = TrainGBM3$bestTune$n.trees,
     continuous.resolution = 100,
     grid.levels = NULL,
     return.grid = FALSE,
     type = "response", main="Marginal effect of age and eosinophils")



#Plot Marginal effect of age and monocytes
plot(TrainGBM3$finalModel,i.var = cbind(var2,age1),
     n.trees = TrainGBM3$bestTune$n.trees,
     continuous.resolution = 100,
     grid.levels = NULL,
     return.grid = FALSE,
     type = "response", main="Marginal effect of age and monocytes")



#Plot Marginal effect of age and lymphocytes
plot(TrainGBM3$finalModel,i.var = cbind(var3,age1),
     n.trees = TrainGBM3$bestTune$n.trees,
     continuous.resolution = 100,
     grid.levels = NULL,
     return.grid = FALSE,
     type = "response", main="Marginal effect of age and lymphocytes")



#Plot ROC curve
predicted=ifelse(predictedGBM3=="X1",1,2)
#convert to numerics
curve1 = roc(response = actuals1, 
             predictor = predicted)
plot(curve1,main="ROC Curve for GBM SMOTE for test dataset",col.main="blue", col.lab="blue",print.thres="best", print.thres.best.method="closest.topleft")
details = paste("AUC GBM SMOTE testing Sample",curve1$auc)
print(details)



########################################################################
# Repeat process but this time analyse all patients who tested positivie for
#COVID and were addmitted to hospital

#Feature engineering
#reset all rows to 1
data1$admittedToHospital=1

#admitted to hospital and tested positive for COV_2_SARS
for (i in 1:NROW(data2)){
  if (data2$patient_addmited_to_regular_ward_1_yes_0_no[i]==2 & data2$sars_cov_2_exam_result[i]==2){data1$admittedToHospital[i]=2}
  if (data2$patient_addmited_to_semi_intensive_unit_1_yes_0_no[i]==2 & data2$sars_cov_2_exam_result[i]==2){data1$admittedToHospital[i]=2}
  if (data2$patient_addmited_to_intensive_care_unit_1_yes_0_no[i]==2 & data2$sars_cov_2_exam_result[i]==2){data1$admittedToHospital[i]=2}
}

#breakup into test and train again 
inTrain=createDataPartition(y=data1$admittedToHospital,times=1,p=p1,list=FALSE)
training=data1[inTrain,]
testing=data1[-inTrain,]

#determine proportions
out1=prop.table(table(training$admittedToHospital))
out2=prop.table(table(testing$admittedToHospital))

Details=paste("Train Imbalance for admittedToHospital==2:",out1)
print(Details)
Details=paste("Test Imbalance for admittedToHospital==2:",out2)
print(Details)

NumTesting1=sum(testing$admittedToHospital==2)
NumTraining1=sum(training$admittedToHospital==2)




##################################################################
## Remove fields used to create target
##################################################################
#creating new dataset for train and test and remove fields that are used to make target feature
TrainNoPred=subset(training, select = -c(patient_addmited_to_regular_ward_1_yes_0_no,patient_addmited_to_semi_intensive_unit_1_yes_0_no,patient_addmited_to_intensive_care_unit_1_yes_0_no,admittedToHospital,sars_cov_2_exam_result,patient_id))
TestNoPred=subset(testing, select = -c(patient_addmited_to_regular_ward_1_yes_0_no,patient_addmited_to_semi_intensive_unit_1_yes_0_no,patient_addmited_to_intensive_care_unit_1_yes_0_no,admittedToHospital,sars_cov_2_exam_result,patient_id))

######################Test are there any issues with data to model
PreProcessCheck2=preProcess(TrainNoPred,method= c("center", "scale","knnImpute"))
PreProcessCheck2





#set the outcome so can be used in the training functions
outcome=as.factor(training$admittedToHospital)
outcomeReg=as.factor(training$admittedToHospital)
a3=factor(outcome, labels = make.names(levels(outcome)))
outcome=a3
outcome2=outcome


#Now get actuals
actuals1=as.factor(testing$admittedToHospital)
a3=factor(actuals1, labels = make.names(levels(actuals1)))
actuals=a3

actualsTrain=as.factor(training$admittedToHospital)
a3=factor(actualsTrain, labels = make.names(levels(actuals)))
actualsTrain=a3


########################################################



Details=paste("Start of regression GBM SMOTE 2 ",Sys.time())
print(Details)
TrainGBM3=train(x=TrainNoPred,y=outcome,method ="gbm",trControl=ctrl3,preProcess = c("center", "scale","knnImpute"))           
Details=paste("End of regression GBM SMOTE 2",Sys.time())
print(Details)

#Print out training result
details=paste("Training result",TrainGBM3$results)
print(details)

#Run prediction on test data
predictedGBM3=data.frame(predict(TrainGBM3,newdata=data.matrix(TestNoPred),type="raw"))
predictionsGBM3=factor(predictedGBM3$predict.TrainGBM3..newdata...data.matrix.TestNoPred...type....raw..)

#confusion matrix
cv=confusionMatrix(predictionsGBM3, actuals, mode="everything")

print("Confusion Matrix Test prediction results for GBM SMote")
print(cv$overall)
print("More Confusion Matrix Test prediction results for GBM SMote") 
print(cv$byClass)

#determine other performance metrics such as R^2
Performance1=data.frame(cbind(actuals,predictionsGBM3))
Performance2=rename(Performance1,obs=actuals, pred=predictionsGBM3)
Performance3=caret::defaultSummary(data=Performance2,  lev=c("1","2"),model = "gbm")
print("Other performance metrics") 
print(Performance3)

#Detetemine influence of certain variables
x1=TrainGBM3$finalModel
x2=data.frame(summary(x1))

#print("Variable influence on model")
#print(x2)

#Look at top 10 most important relative predictors                               
dt1=data.frame(x2[1:10,])
x=factor(dt1$var, levels=dt1$var)
y=dt1$rel.inf

#plot top 10
ggplot() + 
  geom_bar(data = dt1, aes(x=x, y=y), stat="identity",fill="red") +
  coord_flip() + ggtitle("Relative importance of top 10 predictors") +
  labs(y= "Relative Importance", x = "Predictor")

#find the index of variables want to check
for (i in 1:NROW(names1))
{ if (names1[i]=="patient_age_quantile")
{ age1=i}
  if (names1[i]==x[1])
  { var1=i}
  if (names1[i]==x[2])
  { var2=i}
  if (names1[i]==x[3])
  { var3=i}
}                               

#Plot Marginal effect of age and eosinophils
plot(TrainGBM3$finalModel,i.var = cbind(var1,age1),
     n.trees = TrainGBM3$bestTune$n.trees,
     continuous.resolution = 100,
     grid.levels = NULL,
     return.grid = FALSE,
     type = "response", main="Marginal effect of age and eosinophils")


#Plot Marginal effect of age and platelets
plot(TrainGBM3$finalModel,i.var = cbind(var2,age1),
     n.trees = TrainGBM3$bestTune$n.trees,
     continuous.resolution = 100,
     grid.levels = NULL,
     return.grid = FALSE,
     type = "response", main="Marginal effect of age and platelets")



#Plot Marginal effect of age and leukocytes
plot(TrainGBM3$finalModel,i.var = cbind(var3,age1),
     n.trees = TrainGBM3$bestTune$n.trees,
     continuous.resolution = 100,
     grid.levels = NULL,
     return.grid = FALSE,
     type = "response", main="Marginal effect of age and leukocytes")


#Plot ROC curve
predicted=ifelse(predictedGBM3=="X1",1,2)
#convert to numerics
curve1 = roc(response = actuals1, 
             predictor = predicted)
plot(curve1,main="ROC Curve for GBM SMOTE for test dataset",col.main="blue", col.lab="blue",print.thres="best", print.thres.best.method="closest.topleft")
details = paste("AUC GBM SMOTE testing Sample",curve1$auc)
print(details)


