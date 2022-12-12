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
