Hi, I'm trying to reproduce the approach #2 from the Kaggle Porto Seguro winning solution described at https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629

Today I'm still not getting the results from the winner and I'm sharing the code to get some help from the community.

The code is available at https://github.com/lorenzoridolfi/reproduce_porto_winner_solution

The processing workflow is simple. First I prepare the dataset in the file prepare_data.py. After that, I prepare the noise dataset using prepare_data_noise.py. The next step is to process the DAE, done in the file keras_dae.py. Finally, I generate the final result in the keras_final_dae.py

I followed the instructions from the original post and to obtain the DAE I only changed the optimizer from SGD to Adam to get a quicker convergence.

ASSUMPTIONS MADE:

- I'm adding noise before the OHE. I believe shuffling the sparse OHE columns has a very low effect in adding noise.
- I'm considering a column to be a binary column if the column has 3 or less different elements, as the column may have missing values. Only non-binary columns are normalized by rank-gauss.

UPDATE:

Doing some manual parameter adjustment I got a partial gini score of 0.29 in a CV. Now I'm running a hyperopt parameter search, using keras_hyper_dae.py, to discover the best values for lr, l2 reg and dropout. As the winner used C++ and a different NN library, I believe it's normal to have to adjust the parameters. 