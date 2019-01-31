import csv
from matplotlib import pyplot as plt
import sys

# f1 = open('/home/tavish/Desktop/MEGASync/HKUST/FYP/fyp/static_ddpg_tests/lstm/pretrained/pretrained-lstm-eval-rewards.csv', 'r')
# f2 = open('lstm-eval-rewards.csv', 'r')

# reader1 = csv.reader(f1)
# reader2 = csv.reader(f2)


# def get_hits_per_zipf(row):
#     length = len(row)
#     vals = []
#     start = 0
#     end = 1000
#     while end <= length:
#         vals.append(sum(map(float, row[start:end]))/(end-start))
#         start = end
#         end += 1000
#     return vals

# vals1 = get_hits_per_zipf(next(reader1))
# vals2 = get_hits_per_zipf(next(reader2))

# plt.plot(vals1)
# plt.plot(vals2)
# plt.plot([sum(vals1)/len(vals1) for i in vals1])
# plt.show()
# plt.plot([sum(vals2)/len(vals2) for i in vals2])
# plt.legend(['linear','lstm', 'avg-linear', 'avg-lstm'])
# plt.show()

# f1 = open('/home/tavish/Desktop/MEGASync/HKUST/FYP/fyp/static_traditional_tests/lru-hits.csv', 'r')
# reader1 = csv.reader(f1)

# def get_hits_per_zipf(row):
#     length = len(row)
#     print(length)
#     vals = []
#     start = 0
#     end = 1000
#     print(end)
#     while end <= length:
#         vals.append(sum(map(float, row[start:end]))/(end-start))
#         print(vals)
#         start = end
#         end += 1000
#     return vals

# vals = list(map(float, next(reader1)))
# print(vals)
# plt.plot(vals)
# plt.plot([sum(vals)/len(vals) for i in vals])
# plt.legend(['lru','avg-lru'])
# plt.show()


if __name__ == "__main__":
    gru_path = '/home/tavish/Desktop/MEGASync/HKUST/FYP/fyp/static_gru_tests/gru-online-500k.csv'
    lookback_path = '/home/tavish/Desktop/MEGASync/HKUST/FYP/fyp/static_traditional_tests/lookback-hits.csv'
    gru_3_step_path = '/home/tavish/Desktop/MEGASync/HKUST/FYP/fyp/static_gru_tests/m2m-gru-online-500k.csv'
    gru_backtest = '/home/tavish/Desktop/MEGASync/HKUST/FYP/fyp/static_gru_tests/m2m-gru-backtest-500k-1mil.csv'
    dlcpp = '/home/tavish/Desktop/MEGASync/HKUST/FYP/fyp/static_dlcpp_tests/dlcpp-online-500k.csv'

    f1 = open(dlcpp, 'r')
    r = csv.reader(f1)
    vals = list(map(float, next(r)))
    plt.plot(vals)
    mean = sum(vals)/len(vals)
    print(mean)
    plt.plot([mean for i in vals])
    print('a')
    plt.legend(['dlcpp','avg-dlcpp'])
    plt.show()