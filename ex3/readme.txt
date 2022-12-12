Ex3 - api changes and reasonings:

MAIN:
if your comuter supports GPU acceleration and you want to use it change the "USE_ACCELERATION"
variable to True.


API CHANGES:
* in the creation of each model, the default values are the ones spicified for the runs in
  the pdf.

* changed "get_available_device" function to support macOs GPU accelaration, not just cuda.
      - added an import to "platform" library, and the helper function "_running_on_mac()".

* in "create_or_load_slim_w2v" added "w2v_path" argument, to differ between the two dicts,
  (for the average and the sequance)

* in the "DataManager" class, we changed the creation of the w2v dict to check if the file
  alrady exists, and if not - save it.

* in "train_epoch" added an arhument "n" for the epoch iteration number, for the tqdm prints

* in "evaluate" added 2 arguments:
    - "n": epoch iteration number
    - "seq" string of the current data sequance (train / validation / test)

* in "get_predictions_for_data": we changed the input argument, insted of the data_iter,
  we give as input the model's output, we do that to speed up the run times, as anywere
  we need to call this function we alrady have the output calculated.


1. Comparing the results (test accuracy, validation accuracy) for the log-linear model (one-hot and W2V):
    - Which one performs better?
          The W2V preforms better than the One-Hot for the Log-linear model on all the test sets.
    - Possible explanation for the results:
          This could happen because of many reasons like that the vector embedding's lower dimension better encapsulate
          the sentiment of the sentence, and reduce possible "noise" in the sample in the form of words that dos'nt help
          the model with the sentiment analysis.

2. Comparing the log-linear models results with the results of the LSTM model:
    - Which one performs better?
        The LSTM preforms better than the Log-linear model on all the test sets.
    - Possible explanation for the results:
        This probably because the LSTM model is more expressive than the Log-Linear model because of its special
        LSTM cells and the hidden layer structure, therefor can better learn the relations that are relevant for
        the sentiment analysis, furthermore, the LSTM model checks for relations on both directions of the sentence.
        As we can tell from the loss graphs, it seems that the LSTM model actually succeeded in generalizing,
        and the validation loss and acc are improving.

3. Comparing the results that all the models had on the 2 special subsets of sentences:
 #For each subset, state the model that has the highest result (and the lowest result) and provide a possible explanation for these results.

