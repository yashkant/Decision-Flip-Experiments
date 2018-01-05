import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')

def plot_data_graph(l2_test, l2_train, l2_random, l2_random_normal, n, from_cls, to_cls):
    # %matplotlib inline
    t = np.arange(1,n+1, 1)
    plt.plot(t, l2_test[:n], 'r--', t, l2_train[:n],'b--' , t, l2_random[:n], 'y--', l2_random_normal[:n], 'k--')
    blue_patch = mpatches.Patch(color='blue', label='Train Data')
    red_patch = mpatches.Patch(color='red', label='Test Data')
    yellow_patch = mpatches.Patch(color='yellow', label='Random Data')
    black_patch = mpatches.Patch(color='black', label='Random Normal Data')
    plt.legend(handles=[blue_patch, red_patch, yellow_patch, black_patch])
    plt.title("From " + str(from_cls) + " to " + str(to_cls))
    plt.xlabel("Examples")
    plt.ylabel("L2 Norm")

    plt.show()

def plot_data_hist(l2,n,title):
    # %matplotlib inline
    plt.hist(l2,n)
    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Frequency")

    plt.show()

def plot_hists_without_random(l2_test, l2_train, from_cls, to_cls):
    plot_data_hist(l2_test, l2_test.shape[0], "Train Data\n From " + str(from_cls) + " to " + str(to_cls))
    plot_data_hist(l2_train, l2_train.shape[0], "Test Data\n From " + str(from_cls) + " to " + str(to_cls))

def plot_data_graph_without_random(l2_test, l2_train, from_cls, to_cls):
    # %matplotlib inline
    n = min(l2_test.shape[0], l2_train.shape[0])
    t = np.arange(1,n+1 , 1)
    plt.plot(t, l2_test[:n], 'r--', t, l2_train[:n],'b-- ')
    blue_patch = mpatches.Patch(color='blue', label='Train Data')
    red_patch = mpatches.Patch(color='red', label='Test Data')
    yellow_patch = mpatches.Patch(color='yellow', label='Random Data')
    black_patch = mpatches.Patch(color='black', label='Random Normal Data')
    plt.legend(handles=[blue_patch, red_patch, yellow_patch, black_patch])
    plt.title("From " + str(from_cls) + " to " + str(to_cls))
    plt.xlabel("Examples")
    plt.ylabel("L2 Norm")

    plt.show()