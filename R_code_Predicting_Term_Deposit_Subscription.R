library(tidyverse)
library(ROCR)
library(plotROC)
library(pROC)
library(ggpubr)
library(effects)
library(gridExtra)
library(corrplot)
library(dplyr)
library(ggplot2)
library(caret)
library(ROSE)


# Import data for Portuguese Banking Product

bank <- read.csv(here::here("bank-additional", "bank-additional-full.csv"),sep = ";", header = T)
bank.ts <- read.csv(here::here("bank-additional", "bank-additional.csv"),sep = ";", header = T)
# str(bank)
# summary(bank)

# check for missing values (no missingness)
sum(is.na(bank))
apply(is.na(bank), 2, which)


# Changing categorical variables to factors
# levels(bank$job)
bank$job <- as.factor(bank$job)
bank$marital <- as.factor(bank$marital)
bank$education <- as.factor(bank$education)
bank$default <- as.factor(bank$default)
bank$housing <- as.factor(bank$housing)
bank$loan <- as.factor(bank$loan)
bank$contact <- as.factor(bank$contact)
bank$month <- factor(bank$month, levels = tolower(month.abb))
bank$day_of_week <- factor(bank$day_of_week, 
                         levels = c("mon", "tue", "wed", "thu", "fri"))
bank$poutcome <- as.factor(bank$poutcome)
bank$y <- as.factor(bank$y)

## EDA

p1 <- ggplot(bank, aes(x=age)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.2, fill="#FF6666") + theme_minimal()

p2 <- ggplot(bank, aes(x=duration)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.2, fill="#FF6666") + theme_minimal()

p3 <- ggplot(bank, aes(x=campaign)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.2, fill="#FF6666") + theme_minimal()

p4 <- ggplot(bank, aes(x=pdays)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.2, fill="#FF6666") + theme_minimal()

p5 <- ggplot(bank, aes(x=previous)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.2, fill="#FF6666") + theme_minimal()

p6 <- ggplot(bank, aes(x=emp.var.rate)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.2, fill="#FF6666") + theme_minimal()

ggarrange(p1,p2,p3,p4,p5,p6,nrow = 2, ncol = 3)

 
# Dividing dataset into training and test data set (80-20 split)
set.seed(123)
s <- sample(nrow(bank), round(.8*nrow(bank)))
banktrain <- bank[s,]
banktest<- bank[-s,]

# glm on original data
fitlog <- glm(y ~ ., family = "binomial", data = banktrain)
summary(fitlog)

# AIC
fitlog$aic

# Accuracy on validation set
fitted.probabilities <- predict(fitlog,newdata=banktest[-21],type='response')
fitted.results <- factor(ifelse(fitted.probabilities < 0.5, "no", "yes"))

misClasificError <- mean(fitted.results != banktest$y)
misClasificError
print(paste('Accuracy',round((1-misClasificError)*100,2),'%'))

#  confusion matrix
table(banktest$y, fitted.results)

### ROC curve

predobj <- ROCR::prediction(fitted.probabilities,banktest$y)
perf <- performance(predobj,"tpr","fpr")
plot(perf,colorize=TRUE, lwd = 2)

# Area under the curveon validation set: 0.942
auc <- performance(predobj,"auc")
auc(banktest$y,fitted.probabilities)


# Accuracy on test set(10%)
fitted.probabilities_test <- predict(fitlog,newdata=bank.ts[-21],type='response')
fitted.results_test <- factor(ifelse(fitted.probabilities_test < 0.5, "no", "yes"))

misClasificError_test <- mean(fitted.results_test != bank.ts$y)
misClasificError_test
print(paste('Accuracy',round((1-misClasificError_test)*100,2),'%'))

#  confusion matrix
table(bank.ts$y, fitted.results_test)

### ROC curve

predobj_test <- ROCR::prediction(fitted.probabilities_test,bank.ts$y)
perf_test <- performance(predobj_test,"tpr","fpr")
plot(perf_test,colorize=TRUE, lwd = 2)

# Area under the curve ON BANK Test Set (Original 10%): 0.938
auc_test <- performance(predobj_test,"auc")
auc(bank.ts$y,fitted.probabilities_test)


# Marginal Effects

e1 <- plot(allEffects(fitlog), selection = 2,
           axes = list(x=list(rotate = 90)), ylab = "y")

e2 <- plot(allEffects(fitlog), selection = 8,
           axes = list(x=list(rotate = 90)), ylab = "y")

e3<- plot(allEffects(fitlog), selection = 9,
          axes = list(x=list(rotate = 90)), ylab = "y")

e4 <- plot(allEffects(fitlog), selection = 1,
           axes = list(x=list(rotate = 90)), ylab = "y")

grid.arrange(e1,e2,e3,e4, nrow = 2)


#Model 2--------------------------------------------------
# imputing "unknown" in all the variables

# default variable (only 3 cases with "yes" )
table(bank$default)

# education variable (only 18 cases with "illiterate" )
table(bank$education)

# removing these 3 cases with "yes" & 18 cases of illiterate as the number is 
# very less to make significant difference
bankxf <- bank[!(bank$default == "yes") & !(bank$education == "illiterate"),]

# mode function

calculate_mode <- function(x) {
  uniqx <- unique(na.omit(x))
  uniqx[which.max(tabulate(match(x, uniqx)))]
}
sum(is.na(bankxf))
table(bankxf$education)

# replacing "unknown" with most likely event

bankxf[bankxf$job == "unknown","job"] <- as.character(calculate_mode(bankxf$job))
bankxf[bankxf$marital == "unknown","marital"] <- as.character(calculate_mode(bankxf$marital))
bankxf[bankxf$education == "unknown","education"] <- as.character(calculate_mode(bankxf$education))
bankxf[bankxf$housing == "unknown","housing"] <- as.character(calculate_mode(bankxf$housing))
bankxf[bankxf$loan == "unknown","loan"] <- as.character(calculate_mode(bankxf$loan))
as.character(calculate_mode(bankxf$loan))
# replacing 999 in "pdays" variable with -1 so avoid skewness
bankxf[bankxf$pdays == 999,"pdays"] <- -1
table(bankxf$pdays)

# Correlation
#Numeric data columns
nume_col <- c(1, 11:14,16:20)
pearson <- cor(bankxf[,nume_col])

corrplot(pearson, type = "lower")

# Scaling
apply(bankxf[,nume_col],2,sd)
bankxfs <- bankxf %>% mutate(across(where(is.numeric), scale))

apply(bankxfs[,nume_col],2,sd)

# Dividing dataset into training and test data set (80-20 split)
set.seed(123)
sxf <- sample(nrow(bankxfs), round(.8*nrow(bankxfs)))
bankxftrain <- bankxfs[sxf,]
bankxftest<- bankxfs[-sxf,]
sapply(lapply(bankxftrain, unique), length)

table(bankxftrain$y)

# glm after imputation

fit.impute <- glm(y ~ ., family = "binomial", data = bankxftrain)
summary(fit.impute)

# AIC
fit.impute$aic

# Accuracy on test set
pred.imputed <- predict(fit.impute,newdata=bankxftest,type='response')
results.imputed <- factor(ifelse(pred.imputed < 0.5, "no", "yes"))

misClasificError.imputed <- mean(results.imputed != bankxftest$y)
misClasificError.imputed
print(paste('Accuracy',round((1-misClasificError.imputed)*100,2),'%'))


# Outcome Imbalance
# Observe that the dataset predicted outcome (y) is skewed towards 'no' with over 88%.
prop.table(table(bankxf$y))

counts <- table(bankxf$y)
barplot(counts,col = c("darkblue","red"),legend = rownames(counts),
        main = "Term Deposit")

# library(plotrix)
# piepercent<- paste0(toupper(rownames(counts)),"=  ",round(100 * counts / sum(counts), 1), "%")
# pie3D(counts, labels = piepercent, col = c("darkblue","red"),
#   main = "Term Deposit Subscription")
# # legend("right", rownames(counts),cex = 0.7, fill = rainbow(length(counts)))


# Model 3 -------------------------------------------------------------------
#over sampling

data_balanced_over <- ovun.sample(y ~., data=bankxftrain, method="over", N=58438,
                                  seed = 1)$data
table(data_balanced_over$y)

# Model 4 -------------------------------------------------------------------
#under sampling

data_balanced_under <- ovun.sample(y ~., data=bankxftrain, method="under", N=7430,
                                   seed = 1)$data
table(data_balanced_under$y)

# Model 5 -------------------------------------------------------------------
# both
data_balanced_both <- ovun.sample(y ~ ., data = bankxftrain, method = "both",
                                  p=0.5, N=32934, seed = 1)$data
table(data_balanced_both$y)


# build glm models

glm.over <- glm(y ~ ., data = data_balanced_over, family = "binomial")
glm.under <- glm(y ~ ., data = data_balanced_under, family = "binomial")
glm.both <- glm(y ~ ., data = data_balanced_both, family = "binomial")

#make predictions 

pred.over <- predict(glm.over, newdata = bankxftest[-21], type = "response")
pred.under <- predict(glm.under, newdata = bankxftest[-21], type = "response")
pred.both <- predict(glm.both, newdata = bankxftest[-21], type = "response")

# AIC

glm.over$aic
glm.under$aic
glm.both$aic
fitlog$aic
fit.impute$aic

# accuracy

results.over <- factor(ifelse(pred.over < 0.5, "no", "yes"))
misClasificError.over <- mean(results.over != bankxftest$y)
misClasificError.over

results.under <- factor(ifelse(pred.under < 0.5, "no", "yes"))
misClasificError.under <- mean(results.under != bankxftest$y)
misClasificError.under

results.both <- factor(ifelse(pred.both < 0.5, "no", "yes"))
misClasificError.both <- mean(results.both != bankxftest$y)
misClasificError.both


print(paste('Accuracy For Oversampling',round((1-misClasificError.over)*100,2),'%'))

print(paste('Accuracy For Undersampling',round((1-misClasificError.under)*100,2),'%'))

print(paste('Accuracy For Both(Over + Under)',round((1-misClasificError.both)*100,2),'%'))


# AUC on Original data set
roc.curve(banktest$y, fitted.probabilities, col=3)

# Drawing legend.
legend(x = 0.7, y = 0.4,
       legend=c('ORIGINAL','ON IMPUTED','OVERSAMPLING','UNDERSAMPLING', 'BOTH(OVER+UNDER)'),
       col=c(3,4,5,6,7),
       lwd=4, cex =0.7, xpd = TRUE, horiz = FALSE)

# AUC on Imputed dataset
roc.curve(bankxftest$y, pred.imputed, col=4, add.roc = T)


#AUC Oversampling
roc.curve(bankxftest$y, pred.over, col=5, add.roc = T)
# Area under the curve (AUC): 0.943

#AUC Undersampling
roc.curve(bankxftest$y, pred.under, col=6, add.roc = T)
# Area under the curve (AUC): 0.942

#AUC Both
roc.curve(bankxftest$y, pred.both, col=7, add.roc = T)
# Area under the curve (AUC): 0.943


