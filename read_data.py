
import numpy as np


data_dir = 'Training_Out/'

n_experiments = 20


for j in range(n_experiments):
        print("reading experinmet %d" % j)
        
        ## Goal train data
        goal_train_pca=np.load('%s/%s_goal_train_pca.npy' % (data_dir, j), allow_pickle=True)
        goal_train_time=np.load('%s/%s_goal_train_time.npy' % (data_dir, j), allow_pickle=True)
        
        print("Goal train data: ")
        print("goal pca -  %s" % (goal_train_pca.shape,) )
        print("goal time (stacked) -  %s \n" % (np.hstack(goal_train_time).shape,) )        
        
        ## No Play train data
        noplay_train_pca=np.load('%s/%s_noplay_train_pca.npy' % (data_dir, j), allow_pickle=True)
        noplay_train_time=np.load('%s/%s_noplay_train_time.npy' % (data_dir, j), allow_pickle=True)
        
        print("No play train data: ")
        print("noplay pca -  %s" % (noplay_train_pca.shape,) )
        print("noplay time (stacked) -  %s \n" % (np.hstack(noplay_train_time).shape,) )        
 

        ## Goal test data        
        goal_test=np.load('%s/%s_goal_test_pca2.npy' % (data_dir, j), allow_pickle=True)
        goal_test_time=np.load('%s/%s_goal_test_time.npy' % (data_dir, j), allow_pickle=True)
        print("goal test data: ")
        print("goal pca (sacked) -  %s" % (np.vstack(goal_test).shape,) )
        print("goal time (stacked) -  %s \n" % (np.hstack(goal_test_time).shape,) )        

        ## No Play test data
        noplay_test=np.load('%s/%s_noplay_test_pca2.npy' % (data_dir, j), allow_pickle=True)
        noplay_test_time=np.load('%s/%s_noplay_test_time.npy' % (data_dir, j), allow_pickle=True)
        print("No play test data: ")
        print("noplay pca (sacked) -  %s" % (np.vstack(noplay_test).shape,) )
        print("noplay time (stacked) -  %s \n" % (np.hstack(noplay_test_time).shape,) )        

        ## Aim test data
        aim_test=np.load('%s/%s_aim_test_pca2.npy' % (data_dir, j), allow_pickle=True)
        aim_test_time=np.load('%s/%s_aim_time.npy' % (data_dir, j), allow_pickle=True)
        print("Aim test data: ")
        print("aim pca (sacked) -  %s" % (np.vstack(aim_test).shape,) )
        print("aim time (stacked) -  %s \n" % (np.hstack(aim_test_time).shape,) )        
        
        
        
        
