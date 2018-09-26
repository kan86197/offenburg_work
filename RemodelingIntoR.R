library(keras)
library(rJava)
.jinit(".\\JavaHelper")
helper = .jnew("RestructureData")


#copy files from directory dir to directory dest. 
#dir -> directory to get files from, dest -> destination directory
#s -> starting sample, e -> end sample, extension(optional) -> extension of file i.e. ".png"
move_desired_samples <- function(dir, dest, s, e, extension = NULL) {
  if(is.null(extension)){
    fnames = paste0(s:e)
    file.copy(file.path(dir, fnames), file.path(dest))
  } else{
    list_of_files <- list.files(dir, extension)
    file.copy(file.path(dir, list_of_files[s:e]), file.path(dest))
  }
}

preprocess <- function(train_path, val_path, test_path){
  #Directories for moving data.
  #Put the path to the folder which hold the dataset.
  train_set <<- .jcall(helper, "S", "processPath", train_path)
  val_set <<- .jcall(helper, "S", "processPath", val_path)
  test_set <<- .jcall(helper, "S", "processPath", test_path)
  
  train_list <<- list.files(train_set, ".png")
  val_list <<- list.files(val_set, ".png")
  test_list <<- list.files(test_set, ".png")
}

move_samples <- function(num_train, num_val, num_test){
  #Validation folder was manually created, the original data directory does not contain a validation folder
  #Suppress warning due to large amount of warning messages slowing the program
  suppressWarnings({
    
    train_images <- file.path(train_set, "images")
    train_labels <<- file.path(train_set, "labels")
    dir.create(train_images)
    dir.create(train_labels)
    
    val_images <- file.path(val_set, "images")
    val_labels <<- file.path(val_set, "labels")
    dir.create(val_images)
    dir.create(val_labels)
    
    test_images <- file.path(test_set, "images")
    test_labels <<- file.path(test_set, "labels")
    dir.create(test_images)
    dir.create(test_labels)
    
    .jcall(helper, "V", "SetPath", train_labels, train_images)
    .jcall(helper, "V", "deleteOldFiles")
    
    .jcall(helper, "V", "SetPath", val_labels, val_images)
    .jcall(helper, "V", "deleteOldFiles")
    
    .jcall(helper, "V", "SetPath", test_labels, test_images)
    .jcall(helper, "V", "deleteOldFiles")
    
    
    #Change the number to the amount of samples picture you want to use
    move_desired_samples(train_set, train_images, 1, num_train, ".png")
    move_desired_samples(train_set, train_labels, 1, num_train, ".txt")
    move_desired_samples(val_set, val_images, 1, num_val, ".png")
    move_desired_samples(val_set, val_labels, 1, num_val, ".txt")
    move_desired_samples(test_set, test_images, 1, num_test, ".png")
    move_desired_samples(test_set, test_labels, 1, num_test, ".txt")
    
    #invoke java methods to move file into subfolders where keras can understand
    .jcall(helper, "V", "SetPath", train_labels, train_images)
    .jcall(helper, "V", "main", .jarray(list(), "java/lang/String"))
    
    .jcall(helper, "V", "SetPath", val_labels, val_images)
    .jcall(helper, "V", "main", .jarray(list(), "java/lang/String"))
    
    .jcall(helper, "V", "SetPath", test_labels, test_images)
    .jcall(helper, "V", "main", .jarray(list(), "java/lang/String"))
  })
  
  setup_generators()
}




setup_generators <- function(bth_size = 100){
  train_generator <<- flow_images_from_directory(
    train_labels,
    train_datagen,
    batch_size = bth_size,
    color_mode = "grayscale",
    target_size = c(100,100),
    class_mode = "categorical"
  )
  
  validation_generator <<- flow_images_from_directory(
    val_labels,
    validation_datagen,
    target_size = c(100, 100),
    batch_size = bth_size,
    color_mode = "grayscale",
    class_mode = "categorical"
  )
  
  
  test_generator <<- flow_images_from_directory(
    test_labels,
    test_datagen,
    target_size = c(100, 100),
    batch_size = bth_size,
    color_mode = "grayscale",
    class_mode = "categorical"
  )
}


#Model setup, could improve with different normalization/configuration
model_setup <- function(){
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 96, kernel_size = c(8,8), activation = "relu", strides = 2, input_shape = c(100, 100, 1)) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_2d(pool_size = c(3,3), strides = 2) %>%
    layer_conv_2d(filters = 256, kernel_size = c(5,5), activation = "relu", strides = 1) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_2d(pool_size = c(3,3), strides = 2) %>%
    layer_conv_2d(filters = 384, kernel_size = c(3,3), strides = 1, activation = "relu") %>%
    layer_conv_2d(filters = 384, kernel_size = c(3,3), strides = 1, activation = "relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3,3), strides = 1, activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(3,3), strides = 2) %>%
    layer_flatten() %>%
    layer_dense(units = 4096, activation = "relu") %>%
    layer_dense(units = 4096, activation = "relu") %>%
    layer_dense(units = 8, activation = "softmax")
}

#Callback for progress bar
incProg <- R6::R6Class("incProg",
                       inherit = KerasCallback,
                       public = list(
                         losses = NULL,
                         on_epoch_end = function(epoch, logs = list()) {
                           param <- self$params
                           incProgress(1/param$epoch)
                         }
                       )
)

#Callback to get optimizer, and potentially other detail of the model
modDetail <- R6::R6Class("modDetail",
                         inherit = KerasCallback,
                         public = list(
                           losses = NULL,
                           on_train_begin = function(logs = list()) {
                             opti <- self$model$optimizer
                             print(opti)
                           }
                         )
)

incProging <- incProg$new()

findOptimizer <- modDetail$new()


#Setting optimizer function
optimize_with_adam <- function(){
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(lr = 1e-5),
    #optimizer = optimizer_rmsprop(lr = 1e-5),
    #optimizer = "sgd",
    metrics = c("accuracy")
  )
}

optimize_with_rmsprop <- function(){
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    #optimizer = "sgd",
    metrics = c("accuracy")
  )
}

optimize_with_sgd <- function(){
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(lr = 1e-5),
    metrics = c("accuracy")
  )
}

train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen <- image_data_generator(rescale = 1/255)
single_test_datagen <- image_data_generator(rescale = 1/255)


single_test_generator <- ""

#Takes a path to the image, preprocess the image and let the model predict the class of the image.
#text_path is a current idea of show the true class of the image, 
#but not sure how to go about it yet
#Either try to find the label based on the name of the file(Many restriction, prone to error but kind of cool)
#Or let the user put in the path to the label file themself(Probably much more practical)
test_on_single_sample <- function(img_path, text_path = NULL){
  try_path <- img_path
  try_img <- image_load(path = try_path, grayscale = TRUE, target_size = c(100,100))
  try_img <- image_to_array(try_img)
  try_img <- array_reshape(try_img, c(1, 100, 100, 1))
  
  
  if(is.null(text_path)){
    single_test_generator <- flow_images_from_data(x = try_img,
                                                   y = NULL,
                                                   generator = single_test_datagen,
                                                   batch_size = 1,
                                                   shuffle = FALSE)
    
  } else{
    try_label_path <- text_path
    try_label_path <- readtext(try_label_path)
    try_label_path <- as.character(try_label_path)
    try_label_path <- drop(try_label_path)
    try_label_path <- to_categorical(try_label_path)
    try_label_path <- array(try_label_path, dim = c(1,8))
    try_label_path <- array_reshape(try_label_path, c(1,8))
    single_test_generator <- flow_images_from_data(x = try_img,
                                                   y = try_label_path,
                                                   generator = test_datagen,
                                                   batch_size = 1,
                                                   shuffle = FALSE)
    
    results <- evaluate_generator(model, generator = single_test_generator, steps = 20)
    results
  }
  
  result <- predict_generator(model, single_test_generator, steps = 1, verbose = 1)
  result
}


model <- model_setup()
optimize_with_adam()


#Call to start training the model
start_training <- function(num_epoch, num_step_per_epoch, num_validation_steps){
  
  model %>% fit_generator(
    train_generator,
    steps_per_epoch = num_step_per_epoch,
    epochs = num_epoch,
    validation_data = validation_generator,
    validation_steps = num_validation_steps,
    callbacks = list(
      incProging,
      findOptimizer
    )
  )
}

save_weight <- function(){
  save_model_weights_hdf5(model, "prototype_model_weight.h5")
}

load_weight <- function(){
  load_model_weights_hdf5(model, "prototype_model_weight.h5")
}

evaluate_performance <- function(){
  results <- evaluate_generator(model, generator = test_generator, steps = 2)
  results
}
