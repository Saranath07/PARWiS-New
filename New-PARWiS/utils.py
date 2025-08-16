import numpy as np
import argparse

def init(n, precomputed=True, dataset=None, topper=None, k=75):
    """
    Initialize the ground truth score vector and the true topper for a set of n items.

    Arguments:
    precomputed -> Use the precomputed random score vectors when topper has score 100 and rest in range [0, 75].
    dataset -> Use one of the real world datasets' score vectors.
    topper -> The case when the true topper has score x (topper) and the rest have score 100-x.
    """
    scores = k*np.random.rand(n)
    if(precomputed and topper==None and dataset==None):
        if(n == 10):
            scores = np.array([  8.2  , 100.   ,  48.174,  42.926,  52.671,  64.874,  27.607, 62.456,  64.792,  26.614])
        elif(n == 25):
            scores = np.array([ 2.637, 37.092, 35.992,  4.858, 23.636, 30.448, 45.242, 30.948,
                    29.136, 38.14 , 34.818, 29.57 , 30.447, 24.912, 28.451, 74.238,
                    73.627, 26.471, 35.294, 36.311, 70.924, 39.424, 64.263,  3.016, 40.923])
        elif(n == 50):
            scores = np.array([69.72 , 66.96 , 59.993, 53.81 , 44.211, 43.364, 49.359, 12.563, 34.298, 49.05 , 14.32 , 45.721, 60.111, 37.027, 66.176, 47.118,
                    60.457, 38.228, 15.522, 65.783, 65.161, 39.038, 73.657, 52.103, 33.216, 65.346, 36.982, 44.408, 64.675, 40.606,  0.199, 42.142,
                    65.422, 69.91 , 29.524,  8.012, 17.244, 70.475, 57.783, 51.637, 46.264, 46.619,  8.721, 39.563, 53.239, 49.05 , 26.752, 37.529, 60.218, 44.554])
        elif(n == 4):
            scores = np.array([100, 75, 50, 25])
        true_top = np.argmax(scores)
        scores[true_top] = 100
    elif dataset is not None:
        if(dataset == "sushi-A"):
            scores = np.array([0.108, 0.097, 0.128, 0.071, 0.088, 0.103, 0.047, 0.261, 0.069,
                               0.027])
            true_top = np.argmax(scores)
        elif(dataset == "sushi-B"):
            scores = np.array([1.881, 1.795, 2.699, 1.511, 1.305, 1.139, 1.765, 0.973, 3.765,
                               2.439, 1.686, 2.145, 0.756, 1.77 , 1.117, 2.307, 0.639, 0.451,
                               0.761, 3.864, 1.877, 1.404, 1.843, 0.632, 0.584, 1.182, 1.606,
                               1.452, 0.538, 0.59 , 0.635, 1.275, 0.842, 0.577, 1.122, 0.636,
                               1.271, 2.471, 0.704, 0.731, 0.42 , 1.879, 0.397, 0.969, 0.788,
                               0.905, 0.966, 2.19 , 0.655, 0.381, 0.774, 0.626, 0.494, 2.076,
                               0.944, 0.444, 0.37 , 1.124, 0.601, 0.318, 0.412, 2.525, 0.808,
                               0.555, 0.477, 0.993, 0.493, 0.538, 0.511, 0.288, 0.696, 0.774,
                               0.805, 1.287, 0.441, 0.524, 1.52 , 0.43 , 0.361, 2.534, 0.352,
                               0.255, 0.578, 0.421, 0.385, 0.411, 0.197, 0.491, 1.187, 0.248,
                               0.286, 0.598, 0.345, 0.245, 0.352, 0.692, 0.849, 0.122, 0.383,
                               0.571])
            true_top = np.argmax(scores)
        elif(dataset == "jester"):
            scores = np.array([ 5.302,  2.608,  2.433,  0.472,  2.772,  7.328,  1.323,  1.068,
                                1.055,  6.532, 10.656,  8.963,  0.372,  7.321,  0.307,  0.11 ,
                                0.667,  1.037,  2.758,  0.711, 17.756,  4.922,  2.677,  0.357,
                                3.157,  7.077, 44.794,  7.924, 34.773,  1.186, 16.757, 43.819,
                                0.471,  5.247, 39.825, 46.761,  0.495,  7.886,  6.217,  5.37 ,
                                1.328, 13.129,  0.892,  0.274,  5.718,  8.505,  7.988, 12.077,
                               27.967, 72.122,  0.86 ,  1.609, 31.31 , 26.06 ,  3.112, 11.707,
                                0.294,  0.048,  1.143,  1.4  , 17.854, 34.092,  2.492,  0.987,
                               17.773, 24.767,  0.841, 25.08 , 23.626,  3.478,  0.708, 24.848,
                                4.198,  0.378,  1.138, 18.486,  3.4  ,  8.965,  1.59 ,  4.447,
                               10.553,  3.645, 13.133,  3.138,  3.322,  2.343, 10.883, 13.517,
                               55.461,  2.807, 12.763,  5.601, 17.763,  5.498,  5.097,  7.329,
                                8.362,  4.242,  1.711,  4.875])
            true_top = np.argmax(scores)
        elif(dataset == "netflix"):
            scores = np.array([ 5.894,  8.772,  9.296,  6.217, 14.005,  9.955, 16.131, 15.299,
                                6.411,  6.155,  6.723,  5.985,  7.37 ,  9.063,  6.385,  6.472,
                               10.68 , 11.372,  9.071,  8.485, 11.584,  6.859,  6.834, 13.125,
                                6.307,  9.931, 13.283,  6.336, 16.467, 18.942,  9.122, 14.596,
                               18.547,  4.974,  7.404,  5.809,  5.67 ,  7.754,  9.177,  8.828,
                               16.189,  9.042, 10.341,  8.844,  7.88 ,  9.633, 17.507, 15.47 ,
                                9.538,  4.725, 21.544, 11.021,  7.319,  5.836, 12.232, 10.723,
                                7.584,  5.43 , 10.867, 16.049,  9.561,  8.139,  8.244, 20.446,
                               11.139, 16.218,  5.186,  8.513, 10.419, 15.957,  9.785, 15.189,
                               16.128, 16.601, 10.955,  4.988, 12.918,  4.373,  5.949,  9.533,
                                6.196, 13.339,  6.607,  8.9  , 12.408,  5.386, 14.288, 11.934,
                                8.86 ,  5.937,  8.074,  8.903, 19.616,  4.692,  7.146,  9.106,
                               10.532,  4.985,  7.437,  6.345])
            true_top = np.argmax(scores)
        elif(dataset == "movielens"):
            scores = np.array([16.855,  9.437, 13.46 , 11.528,  4.421,  8.016,  6.4  , 10.049,
                                6.864,  6.307, 15.369, 16.257, 12.939,  9.733, 15.676,  7.951,
                                7.759, 13.95 ,  7.741, 14.303,  9.865,  5.201, 11.771, 10.292,
                                8.166, 14.061,  9.76 , 12.778,  4.47 , 10.843,  8.327, 10.749,
                                9.923, 10.597, 18.282,  7.968, 14.049, 12.962, 11.456,  6.22 ,
                                5.636,  8.288,  4.58 , 12.309, 10.319, 10.857, 17.921,  9.03 ,
                                9.089, 10.528, 14.366, 13.306,  6.444, 11.154, 16.828,  7.344,
                               15.5  ,  8.658,  4.16 , 11.31 ,  9.021, 10.165, 10.15 ,  3.925,
                                7.725,  7.267,  9.748, 10.612,  9.822,  9.716, 12.38 ,  6.686,
                                9.462, 10.199, 18.869, 12.215,  3.902,  3.448, 12.601, 13.383,
                                9.508,  7.446, 13.515,  7.633,  5.413, 10.032,  9.946, 11.143,
                                7.094,  8.672, 10.76 , 12.117,  6.717,  6.624,  8.671, 11.173,
                                9.059, 15.585,  2.93 ,  4.288])
            true_top = np.argmax(scores)
    elif topper is not None:
        scores = (100-topper)*np.ones(n)
        true_top = np.random.choice(n, 1)[0]
        scores[true_top] = topper
    else:
        true_top = np.argmax(scores)
        scores[true_top] = 100
    return scores, true_top

def get_ranks(scores):
    """
    Caluculate Ranks of items by assigning the average rank to the items with the same scores.
    """
    ranks = np.zeros(len(scores))
    tmp = [(scores[i], i) for i in range(len(scores))]
    tmp.sort(key=lambda x: x[0], reverse = True)
    (rank,n,i) = (1,1,0)
    while(i < len(scores)):
        j = i
        while((j < len(scores)-1) and (tmp[j][0] == tmp[j+1][0])):
            j += 1
        n = j-i+1
        for j in range(n):
            idx = tmp[i+j][1]
            ranks[idx] = rank + (n-1)/2
        rank += n
        i += n
    return ranks

def get_ranking(n, parameters):
    """
    Get the current ranking of items, their ranks as well as the current topper.
    """
    ranks = get_ranks(parameters)
    ranking = sorted(range(len(parameters)), key = lambda k: parameters[k])
    toppers = np.where(ranks == ranks.min())[0]
    top = np.random.choice(toppers)
    return ranking, ranks, top

def print_metric(name, metric, List=None):
    """
    Print a given metric's mean and standard deviation.
    """
    if List is None:
        metric_mean = np.mean(metric, axis=1)
        metric_stddev = np.sqrt(np.var(metric, axis=1))
        print("\n")
        print(name+"_mean:")
        for i in metric_mean:
            print(i)
        print("\n")
        print(name+"_stddev:")
        for i in metric_stddev:
            print(i)
        print("\n")
    else:
        metric_mean = np.mean(metric, axis=2)
        metric_stddev = np.sqrt(np.var(metric, axis=2))
        for j, l in enumerate(List):
            print("\n")
            print(str(l)+"_"+name+"_mean:")
            for i in metric_mean[j]:
                print(i)
            print("\n")
            print(str(l)+"_"+name+"_stddev:")
            for i in metric_stddev[j]:
                print(i)
            print("\n")

def get_args():
    """
    Get command line arguments. Refer README for instructions.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--n", default=10, type=int, help="number of items N in the dataset")
    parser.add_argument("--experiments", default=10, type=int, help="number of experiments to average the metrics over")
    parser.add_argument("--iterations", default=100, type=int, help="number of iterations the algorithm runs for in a single experiment")
    parser.add_argument("--budget", default=10, type=int, help="the multiple of N you want your max budget to be, metrics will be evaluated in steps of N")
    parser.add_argument("--precomputed", action='store_true', help="use precomputed score vectors for the synthetic case")
    parser.add_argument("--dataset", default=None, help="real-world data you want to evaluate: sushi-A, sushi-B, jester, netflix, movielens")
    parser.add_argument("--toppers", default=None, nargs='+', type=int, help="array of x when the topper has score x and the rest 100-x")
    parser.add_argument("--compute", default="exact", help="The compute method to evaluate pairs for active picking")
    parser.add_argument("--name", default="", help="The log file name to save the csv data")
    parser.add_argument("--save_dir", default=".", help="The save directory of the log file")
    parser.add_argument("--no_save", action="store_true", help="Do not save log file")

    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    return args
