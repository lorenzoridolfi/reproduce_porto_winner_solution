Hi, I'm trying to reproduce the approach #2 from the Kaggle Porto Seguro winning solution described at https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629

Today I'm still not getting the results from the winner and I'm sharing the code to get some help from the community.

The code is available at https://github.com/lorenzoridolfi/reproduce_porto_winner_solution

The processing workflow is simple. First I prepare the dataset in the file prepare_data.py. After that, I prepare the noise dataset using prepare_data_noise.py. The next step is to process the DAE, done in the file keras_dae.py. Finally, I generate the final result in the keras_final_dae.py

I followed the instructions from the original post and to obtain the DAE I only changed the optimizer from SGD to RMSprop to get a quicker convergence.

To obtain the final result, if I use the original network setup (4500-1000-1000), the loss doesn't converge, it stalls at 0.6 approximately. The original setup is in the file keras_final_dae.py. I reduced the network to (450-100-100) in the file keras_final_dae_small.py, I got better results, but still a very bad result.