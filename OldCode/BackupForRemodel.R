library(keras)
library(readtext)
library(OpenImageR)

#copy files from directory dir to directory dest. The images is copy from sample number s to sample number e 
#dir -> directory to get files from, dest -> destination directory
#s -> starting sample, e -> end sample, extension(optional) -> extension of file i.e. ".png"
move_desired_samples <- function(dir, dest, s, e, extension = NULL) {
  if(is.null(extension)){
    fnames = paste0(s:e)
    file.copy(file.path(dir, fnames), file.path(dest))
  } else{
    fnames = paste0(s:e, extension)
    file.copy(file.path(dir, fnames), file.path(dest))
  }
}

#Model setup, could improve with different normalization/configuration, different optimizer
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
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-5),
    #optimizer = "sgd",
    metrics = c("accuracy")
  )
}

train_set <- "C:\\Users\\USER\\Desktop\\Offenburg\\aufgabe_1_data\\train"
train_images <- file.path(train_set, "images")
train_labels <- file.path(train_set, "labels")
dir.create(train_images)
dir.create(train_labels)
test_set <- "C:\\Users\\USER\\Desktop\\Offenburg\\aufgabe_1_data\\test"
test_images <- file.path(test_set, "images")
test_labels <- file.path(test_set, "labels")
dir.create(test_images)
dir.create(test_labels)


move_desired_samples(train_set, train_images, 1, 1000, ".png")

move_desired_samples(train_set, train_labels, 1, 1000, ".txt")

move_desired_samples(test_set, test_images, 1, 3000, ".png")

move_desired_samples(test_set, test_labels, 1, 3000, ".txt")


files <- dir(path = train_images, pattern = ".png", full.names = TRUE)
list_of_images <- sapply(files, image_load, grayscale = TRUE, target_size = c(100, 100))

files <- dir(path = test_images, pattern = ".png", full.names = TRUE)
list_of_test_images <- sapply(files, image_load, grayscale = TRUE, target_size = c(100, 100))

files <- dir(path = train_labels, pattern = ".txt", full.names = TRUE)
list_of_labels <- lapply(files, readtext)
label_matrix = do.call('cbind', lapply(list_of_labels, as.character))

files <- dir(path = test_labels, pattern = ".txt", full.names = TRUE)
list_of_test_labels <- lapply(files, readtext)
test_label_matrix = do.call('cbind', lapply(list_of_test_labels, as.character))


#Preprocessing resulting in rotated images
image_matrix <- sapply(list_of_images, image_to_array)
image_matrix <- lapply(image_matrix, as.numeric)
image_matrix <- do.call('cbind', image_matrix)
image_matrix <- array_reshape(image_matrix, dim =  c(506, 100, 100, 1))

test_image_matrix <- sapply(list_of_test_images, image_to_array)
test_image_matrix <- do.call('cbind', lapply(test_image_matrix, as.numeric))
test_image_matrix <- array_reshape(test_image_matrix, dim =  c(384, 100, 100, 1))

label_matrix <- drop(label_matrix)
label_matrix <- to_categorical(label_matrix)

test_label_matrix <- drop(test_label_matrix)
test_label_matrix <- to_categorical(test_label_matrix)

train_datagen <- image_data_generator(rescale = 1/255)
test_datagen <- image_data_generator(rescale = 1/255)

#failed 
#train_generator <- flow_images_from_data(
#  image_matrix,
#  label_matrix,
#  train_datagen,
#  batch_size = 100
#)


model <- model_setup()

#Directories for flow_images_from_directory
dir_of_image <- "C:\\Users\\USER\\Desktop\\Offenburg\\aufgabe_1_data\\train\\labels"
validation_dir <- "C:\\Users\\USER\\Desktop\\Offenburg\\aufgabe_1_data\\test\\labels"

train_generator <- flow_images_from_directory(
  dir_of_image,
  train_datagen,
  batch_size = 100,
  color_mode = "grayscale",
  target_size = c(100,100),
  class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(100, 100),
  batch_size = 1,
  color_mode = "grayscale",
  class_mode = "categorical"
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 20,
  epochs = 8,
  validation_data = validation_generator,
  validation_steps = 10
)


try_path <- "C:\\Users\\USER\\Desktop\\Offenburg\\aufgabe_1_data\\test\\151.png"
try_label_path <- "C:\\Users\\USER\\Desktop\\Offenburg\\aufgabe_1_data\\test\\151.txt"
try_label_path <- readtext(try_label_path)
try_label_path <- as.character(try_label_path)
try_label_path <- drop(try_label_path)
try_label_path <- to_categorical(try_label_path)
try_label_path <- array(try_label_path, dim = c(1,8))
try_label_path <- array_reshape(try_label_path, c(1,8))




try_img <- image_load(path = try_path, grayscale = TRUE, target_size = c(100,100))
try_img <- image_to_array(try_img)
try_img <- array_reshape(try_img, c(100,100))
plot(as.raster(try_img, max = 255))

try_img <- array_reshape(try_img, c(1, 100, 100, 1))


validation_generator <- flow_images_from_data(x = try_img,
                      y = try_label_path,
                      generator = test_datagen,
                      batch_size = 1,
                      shuffle = FALSE)


result <- predict_generator(model, validation_generator, steps = 1, verbose = 1)
result
plot(result)

results <- evaluate_generator(model, generator = validation_generator, steps = 20)
results

