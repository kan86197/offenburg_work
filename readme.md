# R deep learning project with Keras and ShinyApp

## Introduction
This project involves classifying images (a robot participating in robocup??) might see into one of the 8 classes.

### Functions
* Set up path, parameters and train the model in the ShinyApp UI or
* Load pre-trained weight to use for testing
* Shows result of the training process in a graph and information of the history object returned from fit function
* Using the model to predict/classify a class of a single image. The probability of the image belonging to each class is shown.
* This script and code create new subdirectories to accomodate `flow_from_directory()` function from Keras to be able to read and process the image data.
* Ideas on configuring the models in shinyApp. *update, display model summary and can use different optimizer as well as choose new model.  


## Installations
* [R programming language](https://www.r-project.org/)
* [Python](https://www.python.org/downloads/)
* [Anaconda](https://www.anaconda.com/download/)
* For Keras, use the following commands to install Keras from the RStudio console.  
`install.packages("keras")`  
`library(keras)`  
`install_keras()`  
* This project use rJava to invoke java codes, thus it needs java environment. Download and install [java](https://java.com/en/download/).  
If there are problems when loading rJava, see if the enviroment variable "JAVA_HOME" path is set to your jre folder.  


## Install necessary R packages
* `install.packages("shiny")`
* `install.packages("rJava")`


## Data and setting path

You can download the dataset from [here](https://github.com/kan86197/aufgabe_data)

Setting path to the dataset directory and selecting number of desired samples are now inside the ShinyApp  

The images are changed to 100*100 grayscale using keras generator.

## Tutorials and guides
Aside from several answers from stackoverflow, these are useful documents:
* [Shiny tutorial](https://shiny.rstudio.com/tutorial/), there are video tutorial and written tutorial.
* Many useful information on Keras, [image preprocessing](https://keras.io/preprocessing/image/) and [callbacks](https://keras.rstudio.com/articles/training_callbacks.html) are particularly useful(in my opinion).
* [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) tutorials for writing readme.
* The book Deep Learning with R by François Chollet with J.J. Allaire

## Potential problem
* Have not encounter the bug which create the "%PNG" after chaning the way java code delete old files. Still monitoring to make sure it doesn't reoccur. 

* Each time the Java file is changed RStudio must be restart in order for the code to take effect.


#### Side notes
Have been trying to implement the program using asynchronous programming with promise and future package, not very successful, might be because I don't actually understand async well enough. Async stuff is in "async" branch  

Also, the effort to preprocess the data was a bit of a failure, migth be due to me being unfamiliar with R and Keras and didn't use there functions properly. Still, end up using Keras's flow_image_from_directory() along side some java code which works fine. Old preprocessing codes are in BackupForRemodel.R.  



-------------------------------------------------------------------------------
