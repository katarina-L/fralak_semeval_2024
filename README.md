# fralak_semeval_2024
Source code for the submission of Team Fralak for Semeval 2024 - task 4

How to run this code
 - I found that running everything at once leads to rather unpredictable errors. I opted for making scripts that should be ran separately. Make sure they are in a folder containing a folder ‘models’, a folder ‘pickles’, and a folder ‘outputs’. 
x
1. preprocess.py (preprocesses data)
2. meme_labels_nn.py (train module 1)
3. label_rnn.py (train module 2)
4. gather_data_for_final_module.py (run module 1 and 2 on training data)
5. final_module.py (trains module 3 on the outputs of modules 1 and 2)
6. final_flow.py (usually you can run all of this together; if not, comment out lines in main. Look in main to run predictions on test (comment out ‘prepara_dev_data’ and set ‘test’  to True) or dev (comment out ‘prepare_test_data’ and set ‘test’ to False)


The code is rather messy but this was a time-crunched project
