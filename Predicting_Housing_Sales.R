
##################################################
##################################################
#Intro: This code is created to model Saturn's housing market sale prices for all the properties in the city based on the physical characteristics of the building.
#Author: Elisabeth Jones
#Date created: 05/06/2023
#Notes:
  #Models tested in this code: 
      #linear regression (*highest performing*)
      #linear regression w/ non-sig inputs removed
      #lasso regression
##################################################
##################################################

#read in packages
library('dplyr')
library('caret')
library('mice')
library("rpart.plot")
library('glmnet')
library('ggplot2')

#read in data
setwd('/Users/elisabethjones/Desktop/Saturn_Housing_Project')
df <- read.csv('data_homes.csv', stringsAsFactors = TRUE)

#######################################
#Understanding the data
#######################################
#Notes: 
  #raw descriptives: n=14471, 144 input vars, 1 target var
  #target variable spread looks normal
#######################################

#shape of df
dim(df)

#df structure
str(df)

#Summarize
summary(df)

#Visualize target variable
ggplot(df, aes(SALE_PRICE_LOG10)) + geom_density(fill="darkgoldenrod1")


#######################################
#Wrangling & Pre-processing
#######################################
#Notes:
  #pre-processed descriptives: n=14461, train n = 11569, test n = 2892, 133 input vars, 1 target var
  #Notes on tricky variables (would discuss w/ housing expert irl): 
      # UNIT_PRICE - is not a physical attribute & will be too highly correlated w/ the target variable
      # PRINT_KEY - factor, and would add a lot of noise if dummy coded, removing 
      # NBHD_CODE - after researching, I'm unsure if this is a price or a code given by the county, will leave in & see it's effect
      # Lat_final, Lon_final - longitude and latitude are not useful separately and could cause noise, removing
      # SALE_DATE - unsure on which "sale date" this might represent, removing to avoid noise but could keep in and convert to numeric if requested
      # issue_date	valuation	USED_AS_CD	TOTAL_RENT_AREA WalkIndex	buildings_300m - missing a large amount of data, removing
      # DEPTH_LOG10 FRONT_LOG10 AREA_PARCEL_RECT_LOG10 have a few missing data - keeping in but dropping a total of 10 missing cases from the df 
#######################################

#check for missing data
md.pattern(df, rotate.names = TRUE) 
df.na = which(is.na(df), arr.ind = TRUE)
df.na = data.frame(df.na)
table(df.na) #33 35 36 37 142 143 144
colnames(df)[c(33,35,36,37,142,143,144)]
md.pattern(df[c(33,35,36,37,142,143,144)], rotate.names = TRUE) 

#subset data 
drops <- c('UNIT_PRICE','PRINT_KEY','Lat_final','Lon_final','SALE_DATE','issue_date','valuation','USED_AS_CD','TOTAL_RENT_AREA','WalkIndex','buildings_300m')
df <- df[ , !(names(df) %in% drops)]
#check
dim(df)

#drop na's 
df <- na.omit(df)
#check
dim(df)

#splitting into test/train, using 80% of data for training and 20% for test sets
#seed
set.seed(5723)
#split
inTrain <- createDataPartition(y=df$SALE_PRICE_LOG10, p=.80, list=FALSE)
train <- df[inTrain,]
test <- df[-inTrain,]
#check
dim(train)
dim(test)

#Pre-processing set up 
#get target var index
grep("SALE_PRICE_LOG10", colnames(train))
#copy of df's for pre-processing
train.prep <- train
test.prep <- test
#summaries before pre-processing
summary(train)
summary(test)

#train pre-processing - standardizing and removing any variables with zero variance.
#seed
set.seed(5723)
#pre-process
train.prepmodel <- preProcess(train[,-134], method=c("center", "scale","zv")) #possibly add "corr"
train.prepmodel
train.prep[,-134] <- predict(train.prepmodel, train[,-134])
#check
dim(train.prep)
variable.names(train)

#test pre-processing - standardizing and removing any variables with zero variance.
test.prep[,-134] <- predict(train.prepmodel, test[,-134]) #using train model
#check
dim(test.prep)
variable.names(test.prep)

#summary after pre-processing
summary(train.prep)
summary(test.prep)

#######################################
#Linear Regression
#######################################
#Notes:
  #lm1:
    #RMSE: 0.1515494
    #R2: 0.5436364
    #top vars : NBR_FIREPLACES,FRONT_LOG10,DEPTH_FACTOR 
  #lm2 - removed non-sig vars:
    #RMSE:  0.1535128
    #R2: 0.5317349
    #top vars : GRADE_E,NBR_FIREPLACES,FRONT_LOG10 
#######################################

### Model set-up
#Parameters to control the sampling during parameter tuning and testing, using 10-fold cross validation
ctrl <- trainControl(method = "cv", number=10)
#Performace metrics function
eval_metrics = function(model, df, predictions, target){
  resids = df[,target] - predictions
  resids2 = resids**2
  N = length(predictions)
  r2 = as.character(round(summary(model)$r.squared, 2))
  adj_r2 = as.character(round(summary(model)$adj.r.squared, 2))
  print(paste0("R2: ", adj_r2)) #Adjusted R-squared
  print(paste0("RMSE: ", as.character(round(sqrt(sum(resids2)/N), 2)))) #RMSE
}

### lm1
#linear regression model
lm1 <- train(SALE_PRICE_LOG10 ~ ., data=train.prep, method = "lm", trControl=ctrl)

#What variables are most important
varImp(lm1)

#train prediction
p.train.lm1 <- predict(lm1, newdata=train.prep)

#training model performance
train.perf.lm1 <- data.frame(RMSE = RMSE(p.train.lm1, train.prep$SALE_PRICE_LOG10),
                         Rsquared = R2(p.train.lm1, train.prep$SALE_PRICE_LOG10))
train.perf.lm1

#plotting performance
plot(p.train.lm1,train.prep$SALE_PRICE_LOG10,xlab = "Predicted Values",ylab = "Actual")
abline(a = 0,b = 1,col = "cyan",lwd = 2)   

#test prediction
p.test.lm1 <- predict(lm1, newdata=test.prep)

#test model performance
test.perf.lm1 <- data.frame(RMSE = RMSE(p.test.lm1, test.prep$SALE_PRICE_LOG10),
                             Rsquared = R2(p.test.lm1, test.prep$SALE_PRICE_LOG10))
test.perf.lm1

#plotting performance
plot(p.test.lm1,test.prep$SALE_PRICE_LOG10,xlab = "Predicted Values",ylab = "Actual")
abline(a = 0,b = 1,col = "cyan",lwd = 2)   

### lm2
#pull sig variables by converting summary results to list 
df.sigs = data.frame(summary(lm1)$coef[summary(lm1)$coef[,4] <= .05, 4])
df.sigs <- cbind(sig_vars = rownames(df.sigs), df.sigs)
rownames(df.sigs) <- 1:nrow(df.sigs)
sig.list <- as.list(df.sigs$sig_vars)
sig.list <- sig.list[-1];  #remove intercept 
sig.list <- append(sig.list,'SALE_PRICE_LOG10') #add target var 

#removing insignificant variables
lm2.train <- train.prep[ ,(names(train.prep) %in% sig.list)]
lm2.test <- test.prep[ ,(names(test.prep) %in% sig.list)]

#linear regression model
#seed
set.seed(5723)
#lm2
lm2 <- train(SALE_PRICE_LOG10 ~ ., data=lm2.train, method = "lm", trControl=ctrl)

#What variables are most important
varImp(lm2)

#train prediction
p.train.lm2 <- predict(lm2, newdata=train.prep)

#training model performance
train.perf.lm2 <- data.frame(RMSE = RMSE(p.train.lm2, train.prep$SALE_PRICE_LOG10),
                             Rsquared = R2(p.train.lm2, train.prep$SALE_PRICE_LOG10))
train.perf.lm2

#plotting performance
plot(p.train.lm2,train.prep$SALE_PRICE_LOG10,xlab = "Predicted Values",ylab = "Actual")
abline(a = 0,b = 1,col = "cyan",lwd = 2)   

#test prediction
p.test.lm2 <- predict(lm2, newdata=test.prep)

#test model performance
test.perf.lm2 <- data.frame(RMSE = RMSE(p.test.lm2, test.prep$SALE_PRICE_LOG10),
                            Rsquared = R2(p.test.lm2, test.prep$SALE_PRICE_LOG10))
test.perf.lm2

#plotting performance
plot(p.test.lm2,test.prep$SALE_PRICE_LOG10,xlab = "Predicted Values",ylab = "Actual")
abline(a = 0,b = 1,col = "cyan",lwd = 2)  

#######################################
#Lasso
#######################################
#Notes:
  #lasso:
    #RMSE:  0.1541423
    #R2: 0.5303542
    #top vars : GRADE_E,NBR_FIREPLACES,DEPTH_LOG10
#######################################

### Model set-up

#Get target variable index
grep("SALE_PRICE_LOG10", colnames(train.prep))

#split data for glmnet
y.train = train.prep$SALE_PRICE_LOG10
x.train = data.matrix(train.prep[,-134])
y.test = test.prep$SALE_PRICE_LOG10
x.test = data.matrix(test.prep[,-134])

#Parameters to control the sampling during parameter tuning and testing, using 10-fold cross validation
ctrl <- trainControl(method = "cv", number=10, savePredictions = "all")

### Lasso
#k-fold cross-validation to find optimal lambda values
#seed
set.seed(5723)
#cv
cv_model <- cv.glmnet(x.train, y.train, alpha = 1, nfolds=10)

#produce plot of test MSE by lambda value
plot(cv_model)

#find optimal lambda value that minimizes test MSE
l.min <- cv_model$lambda.min
l.min # 0.0006765171, ~98 vars
l.1se <- cv_model$lambda.1se 
l.1se #0.003962374, ~52 vars

#training lasso model
#seed
set.seed(5723)
#lasso model
lasso <- train(SALE_PRICE_LOG10 ~., data=train.prep, 
                method = "glmnet", 
                tuneGrid=expand.grid(alpha=1,lambda=l.1se),
                trControl=ctrl)

#coefficients
round(coef(lasso$finalModel, lasso$bestTune$lambda),3)

#variable importance
varImp(lasso)

#train prediction
train.predict <- predict(lasso, newdata=train.prep)

#training model performance
train.perf <- data.frame(RMSE = RMSE(train.predict,train.prep$SALE_PRICE_LOG10),
                       Rsquared = R2(train.predict,train.prep$SALE_PRICE_LOG10))
train.perf

#plotting performance
plot(train.predict,train.prep$SALE_PRICE_LOG10,xlab = "Predicted Values",ylab = "Actual")
abline(a = 0,b = 1,col = "cyan",lwd = 2)            

#test prediction
test.predict <- predict(lasso, newdata=test.prep)

#test model performance
test.perf <- data.frame(RMSE = RMSE(test.predict,test.prep$SALE_PRICE_LOG10),
                         Rsquared = R2(test.predict,test.prep$SALE_PRICE_LOG10))

test.perf

#plotting performance
plot(test.predict,test.prep$SALE_PRICE_LOG10,xlab = "Predicted Values",ylab = "Actual")
abline(a = 0,b = 1,col = "cyan",lwd = 2)                         
       
       