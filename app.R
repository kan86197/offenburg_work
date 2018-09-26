library(shiny)
source("RemodelingIntoR.R")


#Currently -> Consist of four tabs: 
#             1) For setting path for the train, validation and test set and selecting number of samples you want. 
#             2) For setting parameters, training the model and display training info afterwards.
#             3) For testing either with the whole test set or a single image. Shows testing results
#             4) Change model(Currently can use vgg16 and resnet50 instantiated with default parameters from keras's functions), 
#                can change optimizer. Display the model summary
ui <- fluidPage(
  titlePanel("Remodel"),
  navbarPage(
    "shinythemes",
    tabPanel("Path and Data",
             sidebarLayout(
               sidebarPanel(
                 textInput("train_path_input", label = h3("Training set path"), value = "C:\\Example\\Path\\To\\Data"),
                 textInput("val_path_input", label = h3("Validation set path"), value = "C:\\Example\\Path\\To\\Data"),
                 textInput("test_path_input", label = h3("Testing set path"), value = "C:\\Example\\Path\\To\\Data"),
                 actionButton("set_path_button", "Set path"),
                 br(),
                 br(),
                 conditionalPanel(
                   condition = "output.data_checker == \"Path is set\"",
                   numericInput("num_train_samples", "Select number of training samples", value = 500),
                   numericInput("num_val_samples", "Select number of validation samples", value = 500),
                   numericInput("num_test_samples", "Select number of testing samples", value = 500),
                   actionButton("select_sample_btn", "Select Sample")
                 )
               ),
               mainPanel(
                 verbatimTextOutput("data_checker"),
                 verbatimTextOutput("trn_num"),
                 verbatimTextOutput("val_num"),
                 verbatimTextOutput("tst_num"),
                 conditionalPanel(
                   condition = "output.data_checker == \"Path is set\"",
                   verbatimTextOutput("sample_checker")
                 )
               )
             )
    ),
    tabPanel("Training",
             sidebarLayout(
               sidebarPanel(
                 numericInput("batch_size", label = h3("Batch Size"), value = 100),
                 numericInput("epoch", label = h3("Number of epoch"), value = 5),
                 numericInput("step_per_epoch", label = h3("Step per epoch"), value = 30),
                 numericInput("validation_step", label = h3("Validation step"), value = 15),
                 actionButton("train_button", label = "Start Training"),
                 actionButton("load_weight_button", label = "Load pre-trained weight")
               ),
               mainPanel(
                 verbatimTextOutput("sanity_check"),
                 verbatimTextOutput("textTrain"),
                 verbatimTextOutput("showsOptimizer"),
                 plotOutput("plotTrain")
               )
             )
    ),
    tabPanel("Testing",
             sidebarLayout(
               sidebarPanel(
                 h2("Predicting the class of a single image"),
                 fileInput("file", label = h3("Select image")),
                 actionButton("single_test_button", label = "Predict"),
                 br(),
                 h2("Test on the test set"),
                 actionButton("test_button", label = "Test")
               ),
               mainPanel(
                 h2("Results", align = "center"),
                 plotOutput("curr_image"),
                 strong("Test result: "),
                 fluidRow(verbatimTextOutput("result", placeholder = TRUE)),
                 br(),
                 strong("File information: "),
                 fluidRow(verbatimTextOutput("file_info", placeholder = TRUE))
               )
             )
    ),
    #These are prototype for ideas of configuring and changing the model on the fly in shiny
    tabPanel("Model Configuration/Selection",
             sidebarLayout(
               sidebarPanel(
                 selectInput("select", label = h3("Choose optimizer"), 
                             choices = list("Adam" = 1, 
                                            "RMSProp Optimizer" = 2, 
                                            "Stochastic Gradient Descent" = 3), 
                             selected = 1),
                 actionButton("recompile", label = "Recompile model"),
                 selectInput("choose_model", label = h3("Choose model"),
                             choices = list("Default application model" = 1,
                                            "vgg16" = 2, 
                                            "resnet50" = 3),
                             selected = 1
                 ),
                 actionButton("change_model", label = "Change model")
               ),
               mainPanel(
                 h3("Model Summary"),
                 verbatimTextOutput("model_summary")
               )
             )
    )
  )
)


server <-  function(input, output, session){
  
  #Setting path and samples part---------------------------------------------------------------------
  
  observeEvent(input$set_path_button, {
    preprocess(input$train_path_input, input$val_path_input, input$test_path_input)
    if(train_set == "Path doesn't exist" || val_set == "Path doesn't exist" || test_set == "Path doesn't exist"){
      output$data_checker <- renderPrint({
        cat("Make sure that all your paths is correct")
      })
    } else{
      output$data_checker <- renderPrint({
        cat("Path is set")
      })
      
      output$sample_checker <- renderPrint({
        cat("Please select your desired number of samples")
      })
    }
    
    output$trn_num <- renderPrint({
      cat("There are", length(train_list), "training samples available")
    })
    output$val_num <- renderPrint({
      cat("There are", length(val_list), "validation samples available")
    })
    output$tst_num <- renderPrint({
      cat("There are", length(test_list), "testing samples available")
    })
    
  })
  
  observeEvent(input$select_sample_btn, {
    tryCatch({
      move_samples(input$num_train_samples, input$num_val_samples, input$num_test_samples)
      output$sample_checker <- renderPrint({
        cat("The data are ready")
      })
    }, warning = function(w){
      output$data_checker <- renderPrint({
        print(w)
      })
    }, error = function(e){
      output$data_checker <- renderPrint({
        print(e)
      })
    }, finally = {
      
    })
  })
  
  
  
  #Parameters setting and training part--------------------------------------------------------------
  
  output$textTrain <- renderPrint({
    cat("Training information: ")
  })
  
  observeEvent(input$load_weight_button, {
    load_weight()
    showNotification("The weight is loaded")
  })
  
  
  observeEvent(input$train_button, {
    output$sanity_check <- renderText({
      cat("The model is currently being train")
    })
    
    
    withProgress(message = 'Training model',
                 detail = 'This may take a while...', value = 0, {
                   
                   tryCatch({
                     setup_generators(input$batch_size)
                     history <- start_training(num_epoch = input$epoch, num_step_per_epoch = input$step_per_epoch, 
                                               num_validation_steps = input$validation_step) 
                   },warning = function(w){
                     print(w)
                   },error = function(e){
                     print(e)
                   } 
                   )
                   
                 })
    
    save_weight()
    
    output$showsOptimizer <- renderPrint({
      findOptimizer$model$optimizer
    })
    
    output$textTrain <- renderPrint({
      history
    })
    
    output$plotTrain <- renderPlot({
      plot(history)
    })
    
    output$sanity_check <- renderText({
      cat("Training is finished")
    })
    
  })
  
  
  
  #Testing part----------------------------------------------------------------------------------------
  observeEvent(input$test_button, {
    tryCatch({
      test_result <- evaluate_performance()
      output$result <- renderPrint(
        test_result
      )
      output$file_info <- renderPrint({
        cat("Testing on the test set")
      })
    }, warning = function(w){
      output$file_info <- renderPrint({
        print(w)
      })
    }, error = function(e){
      output$result <- renderPrint({
        toString(c(e, " Have you set a path to a test set??"))
      })
      
    })
    output$curr_image <- renderPlot({
      NULL
    })
    output$file_info <- renderPrint({
      blank <- ""
      cat(blank)
    })
  })
  
  
  
  #Takes an image and classify it as one of the 8 categories
  observeEvent(input$single_test_button, {
    
    my_file <- input$file
    
    if(is.null(my_file)){
      showNotification("No image selected", type = "error")
    }
    else{
      tryCatch({
        single_test_result <- test_on_single_sample(my_file$datapath)
        output$result <- renderPrint({
          colnames(single_test_result, do.NULL = FALSE)
          colnames(single_test_result) <- c(0,1,2,3,4,5,6,7)
          single_test_result
        })
      }, warning = function(w){
        print(w)
      }, error = function(e){
        output$result <- renderPrint({
          
          cat("Wrong type of files is selected")
        })
      }, finally = {
        
      })
      
      
      output$file_info <- renderPrint({
        str(my_file)
      })
      
      output$curr_image <- renderPlot({
        try_img <- image_load(my_file$datapath, grayscale = TRUE, target_size = c(100,100))
        try_img <- image_to_array(try_img)
        try_img <- array_reshape(try_img, c(100,100))
        plot(as.raster(try_img, max = 255))
      })
    }
  })
  
  
  
  
  
  #Model configuration parts----------------------------------------------------------------
  
  
  observeEvent(input$recompile,{
    switch (input$select,
            "1" = optimize_with_adam(),
            "2" = optimize_with_rmsprop(),
            "3" = optimize_with_sgd()
    )
  })
  
  output$model_summary <- renderPrint({
    sum()
  })
  
  sum <- reactive({
    summary(model)
  })
  
  observeEvent(input$change_model,{
    switch(input$choose_model,
           "1" = {
             model <<- model_setup()
             load_weight()
             optimize_with_adam()
           },
           "2" = model <<- application_vgg16(),
           "3" = model <<- application_resnet50()
    )
    
    output$model_summary <- renderPrint({
      summary(model)
    })
  })
}

shinyApp(ui, server)