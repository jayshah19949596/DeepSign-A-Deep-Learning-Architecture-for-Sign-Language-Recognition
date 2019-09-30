# DeepSign: A Deep Learning Architecture for Sign-Language-Recognition
DeepSign is a new deep-learning architecture which achieves comparable results with limited training data for Sign Language Recognition.

## Paper Link
https://rc.library.uta.edu/uta-ir/bitstream/handle/10106/27803/SHAH-THESIS-2018.pdf?sequence=1&isAllowed=y

## Medium Post
https://medium.com/@jayshah_84248/deepsign-a-deep-learning-pipeline-for-sign-language-recognition-a51a8f116dfc

## core:

- This is the folder where the training code is and the testing code is.  
- It has following files:  
   1. train_auto_encoder_1.py
      - This file consist the code which trains the auto-encoder.
      - This auto encoder is Model-1.
      
   2. train_bi_lstm.py
      - This file consist the code which trains the bi-directional LSTM.
      - This bi-directio2nal LSTM is Model-2.
      - This file also loads the Model-1 and only takes output from enocder of Model-1 to bi-directional LSTM.
      
   3. train_lstm.py
      - This file consist the code which trains the uni-directional LSTM.
      - This uni-directio2nal LSTM is Model-2.
      - This file also loads the Model-1 and only takes output from enocder of Model-1 to uni-directional LSTM.
      
   4. train_vae.py
      - This file consist the code which trains the variational auto-encoder.
      - This auto encoder is Model-1.
      
   5. test_bi_lstm.py
      - This file consists of the code which does inference of bi-directional LSTM.
      - This file loads the freezed model and does predictions on test data.
      
   6. test_lstm.py
      - This file consists of the code which does inference of uni-directional LSTM.
      - This file loads the freezed model and does predictions on test data.

## models:
   1. auto_encoder_1.py
      - This file consist the architecture of auto-encoder.
      - This auto-encoder is Model-1.   
      - It is 10 Layered encoder and 15 layered decoder.
      - The file also defines the cost function and the optimizer.
      - This file is used by `train_auto_encoder_1.py` of `core` module.
      
   2. bi_lstm.py
      - This file consist the architecture of bi-directional lstm.
      - This  bi-directional lstm is Model-2.   
      - The file also defines the cost function and the optimizer.
      - This file is used by `train_bi_lstm.py` of `core` module.
   
   3. lstm.py
      - This file consist the architecture of uni-directional lstm.
      - This  bi-directional lstm is Model-2.   
      - The file also defines the cost function and the optimizer.
      - This file is used by `train_lstm.py` of `core` module.
      
   4. vae.py
      - This file consist the architecture of auto-encoder.
      - This variational auto-encoder is Model-1.   
      - The file also defines the cost function and the optimizer.
      - This file is used by `train_auto_encoder_1.py` of `core` module.
      
## utils:
   1. constants.py
      - This file consists of constant.py
      - It has defined path to data folder and models
      - Only this file needs to be changed if you want to use a custom path with in the project
      
   2. cv_utils.py
      - contains all the `OPENCV` functions that are commonly used by files in the `core` module.
      - functions like reading frames, converting image to black and white, resizinng video frame.
      
   3. os_utils.py
      - contains all the `os` module functions that are commonly used by files in the `core` module.
      - functions like iteratng a directory, creating a folder, joining paths.
      
   4. utility.py
      - contains all the functions that are commonly used by files in the `core` module.
      - functions like `freeze_model`, `prepare_batch_frames_from_bg_data`, `load_a_frozen_model`.
      
      
      

      
