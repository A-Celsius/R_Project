#Load Packages
library(EBImage)
library(keras)

# READ IMAGES
setwd('.\Animal')
pics <- c('Dolphin (1).jpg' ,'Dolphin (2).jpg' ,'Dolphin (3).jpg' ,'Dolphin (4).jpg' ,'Dolphin (5).jpg' ,'Dolphin (6).jpg' ,'Dolphin (7).jpg' ,'Dolphin (8).jpg' ,'Dolphin (9).jpg' ,'Dolphin (10).jpg' ,'Dolphin (11).jpg' ,'Dolphin (12).jpg' ,'Dolphin (13).jpg' ,'Dolphin (14).jpg' ,'Dolphin (15).jpg' ,'Dolphin (16).jpg' ,'Dolphin (17).jpg' ,'Dolphin (18).jpg' ,'Dolphin (19).jpg' ,'Dolphin (20).jpg' ,'Dolphin (21).jpg' ,'Dolphin (22).jpg' ,'Dolphin (23).jpg' ,'Dolphin (24).jpg' ,'Dolphin (25).jpg' ,'Dolphin (26).jpg' ,'Dolphin (27).jpg' ,'Dolphin (28).jpg' ,'Dolphin (29).jpg' ,'Dolphin (30).jpg' ,'Dolphin (31).jpg' ,'Dolphin (32).jpg' ,'Dolphin (33).jpg' ,'Dolphin (34).jpg' ,'Dolphin (35).jpg' ,'Dolphin (36).jpg' ,'Dolphin (37).jpg' ,'Dolphin (38).jpg' ,'Dolphin (39).jpg' ,'Dolphin (40).jpg' ,'Dolphin (41).jpg' ,'Dolphin (42).jpg' ,'Dolphin (43).jpg' ,'Dolphin (44).jpg' ,'Dolphin (45).jpg' ,'Dolphin (46).jpg' ,'Dolphin (47).jpg' ,'Dolphin (48).jpg' ,'Dolphin (49).jpg' ,'Dolphin (50).jpg' ,'Dolphin (51).jpg' ,'Dolphin (52).jpg' ,'Dolphin (53).jpg' ,'Dolphin (54).jpg' ,'Dolphin (55).jpg' ,'Dolphin (56).jpg' ,'Dolphin (57).jpg' ,'Dolphin (58).jpg' ,'Dolphin (59).jpg' ,'Dolphin (60).jpg',
          'Lion (1).jpg' ,'Lion (2).jpg' ,'Lion (3).jpg' ,'Lion (4).jpg' ,'Lion (5).jpg' ,'Lion (6).jpg' ,'Lion (7).jpg' ,'Lion (8).jpg' ,'Lion (9).jpg' ,'Lion (10).jpg' ,'Lion (11).jpg' ,'Lion (12).jpg' ,'Lion (13).jpg' ,'Lion (14).jpg' ,'Lion (15).jpg' ,'Lion (16).jpg' ,'Lion (17).jpg' ,'Lion (18).jpg' ,'Lion (19).jpg' ,'Lion (20).jpg' ,'Lion (21).jpg' ,'Lion (22).jpg' ,'Lion (23).jpg' ,'Lion (24).jpg' ,'Lion (25).jpg' ,'Lion (26).jpg' ,'Lion (27).jpg' ,'Lion (28).jpg' ,'Lion (29).jpg' ,'Lion (30).jpg' ,'Lion (31).jpg' ,'Lion (32).jpg' ,'Lion (33).jpg' ,'Lion (34).jpg' ,'Lion (35).jpg' ,'Lion (36).jpg' ,'Lion (37).jpg' ,'Lion (38).jpg' ,'Lion (39).jpg' ,'Lion (40).jpg' ,'Lion (41).jpg' ,'Lion (42).jpg' ,'Lion (43).jpg' ,'Lion (44).jpg' ,'Lion (45).jpg' ,'Lion (46).jpg' ,'Lion (47).jpg' ,'Lion (48).jpg' ,'Lion (49).jpg' ,'Lion (50).jpg' ,'Lion (51).jpg' ,'Lion (52).jpg' ,'Lion (53).jpg' ,'Lion (54).jpg' ,'Lion (55).jpg' ,'Lion (56).jpg' ,'Lion (57).jpg' ,'Lion (58).jpg' ,'Lion (59).jpg' ,'Lion (60).jpg')
          

myPics <- c()

for (i in 1:120) {
  myPics[[i]] <- readImage(pics[i])
}

# Explore 
# print(myPics[[4]])
# display(myPics[[4]])
# display(myPics[[22]])
# summary(myPics)
# summary(myPics[[4]])
# hist(myPics[[2]])
# str(myPics)

#Resize
for (i in 1:120) {
  myPics[[i]] <- resize(myPics[[i]], 24, 24)
}
str(myPics)

#Reshape
for (i in 1:120) {
  myPics[[i]] <- array_reshape(myPics[[i]], c(24, 24, 3))
}
str(myPics)

# Row Bind
trainx <- NULL;
for(i in 1:50){ trainx <- rbind(trainx,myPics[[i]])}
for(i in 61:110){ trainx <- rbind(trainx,myPics[[i]])}
str(trainx)

testx <- NULL;
for (i in 51:60){ testx <- rbind(testx, myPics[[i]])}
for (i in 111:120){ testx <- rbind(testx, myPics[[i]])}
str(testx)

# 0 = Antelope, 1 = Cat
trainy <- c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
testy <- c(0,0,0,0,0,0,0,0,0,0,
           1,1,1,1,1,1,1,1,1,1)

# One hot encoding
trainlabels <- to_categorical(trainy)
testlabels <- to_categorical(testy)

#Model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 64,activation = 'relu',input_shape = c(1728)) %>%
  layer_dense(units = 32,activation = 'relu') %>%
  layer_dense(units = 16,activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')
summary(model)

#Compile
model %>%
  compile(loss = 'binary_crossentropy',
          optimizer = optimizer_rmsprop(),
          metrics = c('accuracy'))

#Fit Model
history <- model %>%
  fit(trainx,
      trainlabels,
      epochs = 40,
      batch_size = 30,
      validation_split = 0.2)
plot(history)

# Evaluation & Prediction - train data
model %>% evaluate(trainx,trainlabels)
pred1 <- model %>% predict(trainx) %>% k_argmax()
prob <- model %>% predict(trainx) %>% k_argmax()

#Evaluation & Prediction - test data
model %>% evaluate(testx, testlabels)
pred2 <- model  %>% predict(testx) %>% k_argmax()
