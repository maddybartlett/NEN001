import numpy as np
import pickle


#data_dir + str(seed) + '_' + classes[0] + '_train_pca.npy'
#data_dir + str(seed) + '_' + classes[0] + '_train_time.npy'
def prepare_input_data(seed, data_dir):

    classes = ['noplay', 'aimless', 'goal']
#for j in range(n_experiments):
    #seed=j
    goal_train_pca=np.load('%s/%s_goal_train_pca.npy' % (data_dir, seed), allow_pickle=True)
    goal_train_time=np.load('%s/%s_goal_train_time.npy' % (data_dir, seed), allow_pickle=True)

    noplay_train_pca=np.load('%s/%s_noplay_train_pca.npy' % (data_dir, seed), allow_pickle=True)
    noplay_train_time=np.load('%s/%s_noplay_train_time.npy' % (data_dir, seed), allow_pickle=True)

    goal_train=[]
    a=0
    for i in range(len(goal_train_time)):
        b=a+int(len(goal_train_time[i]))
        #b=len(goal_train_time[i])
        goal_train.append(goal_train_pca[a:b])
        #goal_train.append(goal_train_pca[a:(a+b)])
        a=b
        #print(np.asarray(goal_train[-1]).shape[0], goal_train_time[i].shape[0], len(goal_train_time[i]))

    noplay_train=[]
    a=0
    for i in range(len(noplay_train_time)):
        b=a+int(len(noplay_train_time[i]))
        noplay_train.append(noplay_train_pca[a:b])
        a=b

    #    pickle_filename = "Unstacked_Training/%s_goal_train_dict.pkl" % seed
    #    with open(pickle_filename, 'wb') as file:
    #        pickle.dump(goal_train_dict, file)



    goal_test=np.load('%s/%s_goal_test_pca2.npy' % (data_dir, seed), allow_pickle=True)
    goal_test_time=np.load('%s/%s_goal_test_time.npy' % (data_dir, seed), allow_pickle=True)

    noplay_test=np.load('%s/%s_noplay_test_pca2.npy' % (data_dir, seed), allow_pickle=True)
    noplay_test_time=np.load('%s/%s_noplay_test_time.npy' % (data_dir, seed), allow_pickle=True)

    aim_test=np.load('%s/%s_aim_test_pca2.npy' % (data_dir, seed), allow_pickle=True)
    aim_test_time=np.load('%s/%s_aim_time.npy' % (data_dir, seed), allow_pickle=True)

    aim_test_label = np.full(aim_test.shape, (classes.index('aimless')))
    goal_test_label = np.full(goal_test.shape, (classes.index('goal')))
    noplay_test_label = np.full(noplay_test.shape, (classes.index('noplay')))

    noplay_train_label = np.full(np.asarray(noplay_train).shape, (classes.index('noplay')))
    goal_train_label = np.full(np.asarray(goal_train).shape, (classes.index('goal')))
    #np.concatenate
    #np.full(noplay_train.shape, classes.index('noplay'))

    return ((goal_train, goal_train_label, goal_train_time), (noplay_train, noplay_train_label, noplay_train_time)), ((goal_test, goal_test_label, goal_test_time), (noplay_test, noplay_test_label, noplay_test_time), (aim_test, aim_test_label, aim_test_time))   #return train, test datasets => dataset = (class, class_label)


def prepare_input_dataset(seed, data_dir):

    train, test = prepare_input_data(seed, data_dir)

    goal_train, goal_train_label, goal_train_time = train[0]
    noplay_train, noplay_train_label, noplay_train_time = train[1]

    goal_test, goal_test_label, goal_test_time = test[0]
    noplay_test, noplay_test_label, noplay_test_time = test[1]
    aim_test, aim_test_label, aim_test_time = test[2]

    train_data = np.concatenate((np.asarray(goal_train), np.asarray(noplay_train)))
    train_labels = np.concatenate((goal_train_label, noplay_train_label))
    train_times = np.concatenate((np.asarray([len(x) for x in goal_train_time]), np.asarray([len(x) for x in noplay_train_time])))


    test_data = np.concatenate((goal_test, noplay_test, aim_test))
    test_labels = np.concatenate((goal_test_label, noplay_test_label, aim_test_label))
    test_times = np.concatenate((np.asarray([len(x) for x in goal_test_time]), np.asarray([len(x) for x in noplay_test_time]), np.asarray([len(x) for x in aim_test_time])))


    return (train_data, train_labels, train_times), (test_data, test_labels, test_times)


if __name__ == "__main__":

    data_dir = '../nengotest/Training_Out/'
    n_experiments = 20

    for j in range(n_experiments):
        print("reading experinmet %d" % j)
        train, test = prepare_input_data(j, data_dir)

        goal_train, goal_train_label, goal_times = train[0]
        noplay_train, noplay_train_label, noplay_times = train[1]

        goal_test, goal_test_label, _ = test[0]
        noplay_test, noplay_test_label, _ = test[1]
        aim_test, aim_test_label, times = test[2]

        print(aim_test.shape, np.vstack(aim_test).shape)
        print(goal_test.shape, np.vstack(goal_test).shape)
        print(noplay_test.shape, np.vstack(noplay_test).shape)

        print(np.asarray(goal_train).shape, np.vstack(goal_train).shape)
        print(np.asarray(noplay_train).shape, np.vstack(noplay_train).shape)

        print(aim_test_label.shape, goal_test_label.shape, noplay_test_label.shape)
        print(goal_train_label.shape, noplay_train_label.shape)

        ((train_data, train_labels, times2), (eval_data, eval_labels, _)) = prepare_input_dataset(j, data_dir)

        print(train_data.shape, train_labels.shape)
        print(eval_data.shape, eval_labels.shape)

        #print(train_data[1].shape)


    #print(type(train_data[1][1][1]))
    #print(type(train_labels[1]))

    #print(times2)

    #print(goal_train[-1])
    #print(goal_train[-2])
    #print(goal_train[-3].shape)

    #for i, data in enumerate(np.asarray(goal_train)):
    #    print(i, data.shape)
    #for i, time in enumerate([len(x) for x in goal_times]):
    #    print(i,time)


    for i, data in enumerate(np.asarray(noplay_train)):
        print(i, data.shape)
    for i, time in enumerate([len(x) for x in noplay_times]):
        print(i,time)


    for i, data in enumerate(train_data):
        print(i, data.shape)
