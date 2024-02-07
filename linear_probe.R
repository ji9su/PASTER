
library(data.table)
library(magrittr)
library(glmnet)

# 0. pathway & settings

label <- c("left ventricular dysfunction")

n_fold <- 4
used_method <- 'lambda.1se'

data_path <- './data/example_data(img_feat).csv'


# 1. Load data

data <- read.csv(data_path, check.names = FALSE)
x_name <- colnames(data)[grepl('img_feat', colnames(data))]


# 2. Data aug

train_data <- data[data[,'DATASET'] %in% c('train'), c('UID', label, x_name)]

pos_1 <- which(train_data[,label] %in% 1)
pos_0 <- which(train_data[,label] %in% 0)

if (length(pos_1) >= n_fold) {pos_1 <- pos_1[1:(floor(length(pos_1) / n_fold) * n_fold)]}
if (length(pos_0) >= n_fold) {pos_0 <- pos_0[1:(floor(length(pos_0) / n_fold) * n_fold)]}

pos_1 <- pos_1 %>% rep(., ceiling(length(pos_0) / length(pos_1)))
pos_0 <- pos_0

train_X <- train_data[c(pos_1, pos_0),-1:-2,drop=FALSE] %>% as.matrix()
train_Y <- train_data[c(pos_1, pos_0),label,drop=FALSE] %>% as.matrix()

train_foldid <- rep(seq(n_fold), nrow(train_X))[1:nrow(train_X)]


# 3. Search parameter

alpha_range <- seq(0, 1, by = 0.1)
lambda_range <- exp(seq(-9, -1, length.out = 45))

param_list <- list()
sub_model_list <- list()

for (m in alpha_range) {
  
  t1 <- Sys.time()
  message('Alpha = ', m)
  
  cv_fit <- cv.glmnet(x = train_X, y = train_Y, family = 'binomial', lambda = lambda_range, alpha = m, type.measure = "auc", foldid = train_foldid, parallel = TRUE)
  param_list[[length(param_list) + 1]] <- data.frame(alpha = m, auc = max(cv_fit[['cvm']]), lambda = cv_fit[[used_method]])
  sub_model_list[[length(sub_model_list) + 1]] <- cv_fit
  
  print(param_list[[length(param_list)]])
  message('time = ', formatC(as.numeric(Sys.time() - t1, unit = 'mins'), format = 'f', 1), ' mins')
  
}

param_data <- do.call('rbind', param_list)
best_model <- sub_model_list[[which.max(param_data[,'auc'])]]

model_coef <- glmnet:::coef.glmnet(best_model, s = used_method) %>% as.matrix()
model_coef <- as.data.frame(model_coef)


# Write out

write.csv(model_coef, file = './result/model_coef.csv')

