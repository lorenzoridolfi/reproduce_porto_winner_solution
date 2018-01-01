Hi, I'm trying to reproduce the approach #2 from the Kaggle Porto Seguro winning solution described at https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629

Today I'm still not getting the results from the winner and I'm sharing the code to get some help from the community. This project is an ongoing development and I'm constantly publishing the progresses made. I'd kindly ask any help on reviewing the code to find any bug or better approaches to the solution.

The processing workflow is simple. First I prepare the dataset in the file prepare_data.py. After that, I prepare the noise dataset using prepare_data_noise.py. The next step is to process the DAE, done in the file keras_dae.py. Finally, I will generate the final result in the keras_final_dae.py. This last program is not tuned yet and is not returning the results obtained by the winner of the competition.

I followed the instructions from the original post and, to obtain the DAE, I only changed the optimizer from SGD to Adam to get a quicker convergence. For the final NN, as the results are not ok yet, I'll work on the hyperparameter tuning to improve the score.

ASSUMPTIONS MADE:

- I'm adding noise before the OHE. I believe shuffling the sparse OHE columns has a very low effect in adding noise.
- I'm considering a column to be a binary column if the column has 3 or less different elements, as the column may have missing values. Only non-binary columns are normalized by rank-gauss.

This project has many contributions gattered from the Porto Seguro discussion board on Kaggle. I thank all the contributors who shared their own code to all participants.

UPDATE:

- In the first run, to generate the DAE, I used RMSprop and used early-stopping with a patience of 20 iterations over improvements on loss. It stopped around 300 iterations. To be sure, I'm running the DAE again, now with Adam and a patience of 100 iterations. After a few iterations, it's clear that will be an improvement.

- To fine tune the LR of the network training the DAE, I made the program "keras_refine_dae.py" to play with different learning rates. As Keras was already automatically saving the network with better performance (lower loss), the learning process could be interrupted anytime to resume with a different learning rate. At the end, I was adjusting the learning rate manually, going as low as 0.00000001.

- The new DAE is already generated. I got a mse error of 0.0085972559289 in the NN. Now I'm performing the hyper-parameter search for the second NN.

