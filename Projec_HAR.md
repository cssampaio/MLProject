---
title: "Practical Machine Learning Project"
author: "cssampaio"
---
##  Practical Machine Learning Project
cssampaio

### Introdution

This report analises data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants in order to identify how well barbell lifts are performed. The data is extracted from Weight Lifting Exercises Dataset for Human Activity Recognition. More information can be found in the source below.

Source: http://groupware.les.inf.puc-rio.br/har

The Weight Lifting Exercises Dataset contains the "classe" variable in the training set, which classifies the execution of the exercise:

    Class A = exactly according to the specification
    Class B = throwing the elbows to the front
    Class C = lifting the dumbbell only halfway
    Class D = lowering the dumbbell only halfway
    Class E = throwing the hips to the front

This document shows the selection of the predictors and the analysis of some machine learning predictive models in the training set. The machine learning predictive model with best accuracy is, then, applied to the testing set to obtain the classification of the outcome.

### Initialization

Loading useful libraries.

```r
library(caret)
library(rpart)
library(randomForest)
library(gbm)
library(plyr)
```

Setting seed for reproducibility.

```r
set.seed(1234)
```

Loading data.

```r
if (!file.exists("pml-training.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                  "pml-training.csv")
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                  "pml-testing.csv")
}

csvtest <- "pml-testing.csv"
testData <- read.csv(csvtest, na.strings = c("NA", ""), header = TRUE)

csvtrain <- "pml-training.csv"
trainData <- read.csv(csvtrain, na.strings = c("NA", ""), header = TRUE)
```

### Exploratory analysis

Exploring the data set.

```r
# str(trainData)
dim(trainData)
```

```
## [1] 19622   160
```

```r
trainData[c(1,24),]
```

```
##     X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1   1  carlitos           1323084231               788290 05/12/2011 11:23
## 24 24  carlitos           1323084232               996313 05/12/2011 11:23
##    new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
## 1          no         11      1.41       8.07    -94.4                3
## 24        yes         12      1.51       8.10    -94.4                3
##    kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
## 1                <NA>                <NA>              <NA>
## 24           5.587755             #DIV/0!           #DIV/0!
##    skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt
## 1                <NA>                 <NA>              <NA>            NA
## 24           2.713152              #DIV/0!           #DIV/0!         -94.3
##    max_picth_belt max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt
## 1              NA         <NA>            NA             NA         <NA>
## 24              3          5.6         -94.4              3          5.6
##    amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
## 1                   NA                   NA               <NA>
## 24                 0.1                    0             0.0000
##    var_total_accel_belt avg_roll_belt stddev_roll_belt var_roll_belt
## 1                    NA            NA               NA            NA
## 24                    0           1.5              0.1             0
##    avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt
## 1              NA                NA             NA           NA
## 24            8.1                 0              0        -94.4
##    stddev_yaw_belt var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z
## 1               NA           NA         0.00            0        -0.02
## 24               0            0         0.02            0        -0.02
##    accel_belt_x accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y
## 1           -21            4           22            -3           599
## 24          -20            4           22            -3           601
##    magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm
## 1           -313     -128      22.5    -161              34            NA
## 24          -318     -129      20.7    -161              34             0
##    avg_roll_arm stddev_roll_arm var_roll_arm avg_pitch_arm
## 1            NA              NA           NA            NA
## 24    -128.4898          0.5051       0.2551       21.4327
##    stddev_pitch_arm var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm
## 1                NA            NA          NA             NA          NA
## 24           0.4836        0.2339        -161              0           0
##    gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z
## 1         0.00           0       -0.02        -288         109        -123
## 24       -0.02           0       -0.02        -289         110        -125
##    magnet_arm_x magnet_arm_y magnet_arm_z kurtosis_roll_arm
## 1          -368          337          516              <NA>
## 24         -374          350          516          -1.05825
##    kurtosis_picth_arm kurtosis_yaw_arm skewness_roll_arm
## 1                <NA>             <NA>              <NA>
## 24            #DIV/0!          #DIV/0!           0.13832
##    skewness_pitch_arm skewness_yaw_arm max_roll_arm max_picth_arm
## 1                <NA>             <NA>           NA            NA
## 24            #DIV/0!          #DIV/0!         22.3          -161
##    max_yaw_arm min_roll_arm min_pitch_arm min_yaw_arm amplitude_roll_arm
## 1           NA           NA            NA          NA                 NA
## 24          34         20.7          -161          34                1.6
##    amplitude_pitch_arm amplitude_yaw_arm roll_dumbbell pitch_dumbbell
## 1                   NA                NA      13.05217        -70.494
## 24                   0                 0      13.00000        -70.700
##    yaw_dumbbell kurtosis_roll_dumbbell kurtosis_picth_dumbbell
## 1     -84.87394                   <NA>                    <NA>
## 24    -84.70000                -0.6209                 -0.6149
##    kurtosis_yaw_dumbbell skewness_roll_dumbbell skewness_pitch_dumbbell
## 1                   <NA>                   <NA>                    <NA>
## 24               #DIV/0!                -0.0960                  0.1049
##    skewness_yaw_dumbbell max_roll_dumbbell max_picth_dumbbell
## 1                   <NA>                NA                 NA
## 24               #DIV/0!             -70.1              -84.3
##    max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell min_yaw_dumbbell
## 1              <NA>                NA                 NA             <NA>
## 24             -0.6               -71              -85.3             -0.6
##    amplitude_roll_dumbbell amplitude_pitch_dumbbell amplitude_yaw_dumbbell
## 1                       NA                       NA                   <NA>
## 24                    0.96                     1.02                   0.00
##    total_accel_dumbbell var_accel_dumbbell avg_roll_dumbbell
## 1                    37                 NA                NA
## 24                   37             0.0204           13.1942
##    stddev_roll_dumbbell var_roll_dumbbell avg_pitch_dumbbell
## 1                    NA                NA                 NA
## 24               0.1811            0.0328           -70.5253
##    stddev_pitch_dumbbell var_pitch_dumbbell avg_yaw_dumbbell
## 1                     NA                 NA               NA
## 24                0.2384             0.0568         -84.8053
##    stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x gyros_dumbbell_y
## 1                   NA               NA                0            -0.02
## 24               0.256           0.0656                0            -0.02
##    gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
## 1                 0             -234               47             -271
## 24                0             -235               47             -271
##    magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z roll_forearm
## 1               -559               293               -65         28.4
## 24              -558               291               -71         27.1
##    pitch_forearm yaw_forearm kurtosis_roll_forearm kurtosis_picth_forearm
## 1          -63.9        -153                  <NA>                   <NA>
## 24         -63.7        -151               -0.3680                -2.0402
##    kurtosis_yaw_forearm skewness_roll_forearm skewness_pitch_forearm
## 1                  <NA>                  <NA>                   <NA>
## 24              #DIV/0!                0.2113                -0.2117
##    skewness_yaw_forearm max_roll_forearm max_picth_forearm max_yaw_forearm
## 1                  <NA>               NA                NA            <NA>
## 24              #DIV/0!            -63.7              -151            -0.4
##    min_roll_forearm min_pitch_forearm min_yaw_forearm
## 1                NA                NA            <NA>
## 24              -64              -152            -0.4
##    amplitude_roll_forearm amplitude_pitch_forearm amplitude_yaw_forearm
## 1                      NA                      NA                  <NA>
## 24                    0.3                       1                  0.00
##    total_accel_forearm var_accel_forearm avg_roll_forearm
## 1                   36                NA               NA
## 24                  36                 0         27.40204
##    stddev_roll_forearm var_roll_forearm avg_pitch_forearm
## 1                   NA               NA                NA
## 24             0.45893          0.21062         -63.89388
##    stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
## 1                    NA                NA              NA
## 24              0.07474           0.00559        -151.449
##    stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
## 1                  NA              NA            0.03            0.00
## 24            0.50254         0.25255            0.03           -0.03
##    gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
## 1            -0.02             192             203            -215
## 24            0.00             193             203            -213
##    magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
## 1               -17              654              476      A
## 24              -11              661              470      A
```

The data set has 160 variables, but some variables do not work as predictors for the "classe" outcome:
    
    * Informative variables, such as "user_name" that identifies the participant.
    * Statistics variables, such as "skewness" only appear when the variable "new_window" = "yes".

Selected predictors to compute the outcome.

```r
grep("^accel|^gyros|^magnet|^num|^pitch|^roll|^total|^yaw", names(trainData), value = TRUE)
```

```
##  [1] "num_window"           "roll_belt"            "pitch_belt"          
##  [4] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
##  [7] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [10] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [13] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [16] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [19] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [22] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [25] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [28] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [31] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [34] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [37] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [40] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [43] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [46] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [49] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [52] "magnet_forearm_y"     "magnet_forearm_z"
```

### Cleaning data set

Based on the previous check, the training data set is cleaned to keep only the valid predictors and the outcome. Also, the observations correponding to "new_window" = "yes" are removed.

```r
# Creating data set with valid predictors and outcome
trainClean <- trainData[,grepl("^accel|^gyros|^magnet|^new|^num|^pitch|^roll|^total|^yaw|^classe", names(trainData))]

# Removing new_window = "yes" observations
trainClean <- trainClean[trainClean$new_window == "no",]
# Removing new_window column
trainClean <- trainClean[,-c(trainClean$new_window)]

# str(trainClean)
dim(trainClean)
```

```
## [1] 19216    54
```

```r
sum(is.na(trainClean))
```

```
## [1] 0
```

Similarly, the testing data set is cleaned.

```r
testClean <- testData[,grepl("^accel|^gyros|^magnet|^num|^pitch|^roll|^total|^yaw|^problem", names(testData))]

# str(testClean)
dim(testClean)
```

```
## [1] 20 54
```

### Machine Learning Predictive Models

Dividing training set for validation.

```r
inTrain = createDataPartition(trainClean$classe, p = 0.6, list = FALSE)
training = trainClean[inTrain,]
testing = trainClean[-inTrain,]
```

Applying predictive models.

The default resampling for the train function in caret package uses bootstrapping. Using trControl parameter to set 10-fold cross-validation for resampling.

```r
# 10-fold cross-validation
tc <- trainControl("cv", 10)
```

#### Decision Tree


```r
# Model fit
modFit1 <- train(classe ~ ., data = training, method = "rpart", trControl = tc)
# modFit1
# fancyRpartPlot(modFit1$finalModel)

# Model accuracy
predictions1 <- predict(modFit1, newdata = testing)
cm1 <- confusionMatrix(predictions1, testing$classe)
cm1
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1966  660  629  584  192
##          B   40  475   49  231  181
##          C  146  352  662  443  359
##          D    0    0    0    0    0
##          E   36    0    0    0  679
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4922         
##                  95% CI : (0.481, 0.5034)
##     No Information Rate : 0.2847         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3357         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8985  0.31944  0.49403   0.0000  0.48122
## Specificity            0.6243  0.91915  0.79508   1.0000  0.99426
## Pos Pred Value         0.4877  0.48668  0.33741      NaN  0.94965
## Neg Pred Value         0.9392  0.84914  0.88151   0.8363  0.89496
## Prevalence             0.2847  0.19352  0.17439   0.1637  0.18363
## Detection Rate         0.2559  0.06182  0.08615   0.0000  0.08837
## Detection Prevalence   0.5246  0.12702  0.25534   0.0000  0.09305
## Balanced Accuracy      0.7614  0.61929  0.64456   0.5000  0.73774
```

Trying to improve model fit accuracy by setting tuneLength.

```r
# Model fit
modFit2 <- train(classe ~ ., data = training, method = "rpart", trControl = tc, tuneLength = 10)
# modFit2
# fancyRpartPlot(modFit2$finalModel)

# Model accuracy
predictions2 <- predict(modFit2, testing)
cm2 <- confusionMatrix(predictions2, testing$classe)
cm2
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1794  253   26  131   82
##          B   40  575   49   24   50
##          C   97  182 1017  181  166
##          D  200  330  172  854  149
##          E   57  147   76   68  964
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6773          
##                  95% CI : (0.6667, 0.6877)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.5924          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8199  0.38668   0.7590   0.6789   0.6832
## Specificity            0.9105  0.97370   0.9013   0.8676   0.9445
## Pos Pred Value         0.7848  0.77913   0.6190   0.5009   0.7348
## Neg Pred Value         0.9270  0.86870   0.9465   0.9324   0.9298
## Prevalence             0.2847  0.19352   0.1744   0.1637   0.1836
## Detection Rate         0.2335  0.07483   0.1324   0.1111   0.1255
## Detection Prevalence   0.2975  0.09604   0.2138   0.2219   0.1707
## Balanced Accuracy      0.8652  0.68019   0.8301   0.7732   0.8139
```

```r
# Out of sample error
ose2 <- 1- cm2$overall['Accuracy']
attr(ose2,"names") <- "ose2"
ose2
```

```
##      ose2 
## 0.3227486
```

```r
# Confusion Matrix plot
results <- data.frame(pred = predictions2, obs = testing$classe)
p <- ggplot(results, aes(x = pred, y = obs, color = factor(pred)))
p <- p +
    labs(title = "Decision Tree Confusion Matrix",
        x = "Predictions", y = "Observations") + 
    scale_color_discrete(name="Predictions") +
    geom_jitter(position = position_jitter(width = 0.25, height = 0.25))
p
```

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-11-1.png) 

Some improvement, but still low accuracy.

#### Random Forest


```r
# Model fit
modFit3 <- train(classe ~ ., data = training, method = "rf", trControl = tc)
# modFit3

# Model accuracy
predictions3 <- predict(modFit3, testing)
cm3 <- confusionMatrix(predictions3, testing$classe)
cm3
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2188    4    0    0    0
##          B    0 1480    6    0    0
##          C    0    3 1334    6    0
##          D    0    0    0 1252    5
##          E    0    0    0    0 1406
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9969         
##                  95% CI : (0.9954, 0.998)
##     No Information Rate : 0.2847         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.996          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9953   0.9955   0.9952   0.9965
## Specificity            0.9993   0.9990   0.9986   0.9992   1.0000
## Pos Pred Value         0.9982   0.9960   0.9933   0.9960   1.0000
## Neg Pred Value         1.0000   0.9989   0.9991   0.9991   0.9992
## Prevalence             0.2847   0.1935   0.1744   0.1637   0.1836
## Detection Rate         0.2847   0.1926   0.1736   0.1629   0.1830
## Detection Prevalence   0.2853   0.1934   0.1748   0.1636   0.1830
## Balanced Accuracy      0.9996   0.9972   0.9971   0.9972   0.9982
```

```r
# Out of sample error
ose3 <- 1- cm3$overall['Accuracy']
attr(ose3,"names") <- "ose3"
ose3
```

```
##        ose3 
## 0.003123373
```

```r
# Confusion Matrix plot
results <- data.frame(pred = predictions3, obs = testing$classe)
p <- ggplot(results, aes(x = pred, y = obs, color = factor(pred)))
p <- p +
    labs(title = "Random Forest Confusion Matrix",
        x = "Predictions", y = "Observations") + 
    scale_color_discrete(name="Predictions") +
    geom_jitter(position = position_jitter(width = 0.25, height = 0.25))
p
```

![plot of chunk unnamed-chunk-12](figure/unnamed-chunk-12-1.png) 

Very high accuracy.

#### Boosting


```r
# Model fit
modFit4 <- train(classe ~ ., data = training, method = "gbm", trControl = tc, verbose = FALSE)
# modFit4

# Model accuracy
predictions4 <- predict(modFit4, testing)
cm4 <- confusionMatrix(predictions4, testing$classe)
cm4
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2187   18    0    0    1
##          B    1 1457   14    3    7
##          C    0   10 1323   16    2
##          D    0    2    3 1238   15
##          E    0    0    0    1 1386
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9879          
##                  95% CI : (0.9852, 0.9902)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9847          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9995   0.9798   0.9873   0.9841   0.9823
## Specificity            0.9965   0.9960   0.9956   0.9969   0.9998
## Pos Pred Value         0.9914   0.9831   0.9793   0.9841   0.9993
## Neg Pred Value         0.9998   0.9952   0.9973   0.9969   0.9960
## Prevalence             0.2847   0.1935   0.1744   0.1637   0.1836
## Detection Rate         0.2846   0.1896   0.1722   0.1611   0.1804
## Detection Prevalence   0.2871   0.1929   0.1758   0.1637   0.1805
## Balanced Accuracy      0.9980   0.9879   0.9914   0.9905   0.9911
```

```r
# Out of sample error
ose4 <- 1 - cm4$overall['Accuracy']
attr(ose4,"names") <- "ose4"
ose4
```

```
##       ose4 
## 0.01210307
```

```r
# Confusion Matrix plot
results <- data.frame(pred = predictions4, obs = testing$classe)
p <- ggplot(results, aes(x = pred, y = obs, color = factor(pred)))
p <- p +
    labs(title = "Boosting Confusion Matrix",
        x = "Predictions", y = "Observations") + 
    scale_color_discrete(name="Predictions") +
    geom_jitter(position = position_jitter(width = 0.25, height = 0.25))
p
```

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13-1.png) 

Very high accuracy.

### Conclusion

Both Random Forest and Boosting provide very high accuracy, with Random Forest having better performance than Boosting.

### Teste Case Predictions

Checking outcome predictions for 20 different test cases.

```r
tcpredictions3 <- predict(modFit3, testClean)
tcpredictions3
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
tcpredictions4 <- predict(modFit4, testClean)
tcpredictions4
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
# Comparing test case predictions
all(tcpredictions3 == tcpredictions4)
```

```
## [1] TRUE
```

Both Random Forest and Boosting give the same outcome predictions for 20 different test cases.

Generating test case prediction files.

```r
# Generating prediction files
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(tcpredictions3)
```
