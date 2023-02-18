"""
Benchmarking of Pandas
Perform below operations on different data sizes.
SQL 1: select sum(close) from data
SQL 2: sort close column
"""

import pandas as pd
import glob
import os
import time
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', help='Path oto the csv file folder')
args = parser.parse_args()

path = args.path
all_files = glob.glob(os.path.join(path, "*.csv"))

# Benchmarking data for pandas
row_list = []
for n in [2, 4, 6, 8, 10, 15, 20]:
    # In my case each csv file is approximately 630 MB so each file s 0.615 GB
    # Dividing by 0.615 will give the number of files required to make n GB
    n = round(n / 0.615)
    print("Number of files to process: ", n)

    size = 0
    dict = {}
    df = pd.DataFrame()
    for f in all_files[:n]:
        file_size = os.path.getsize(f)
        size = size + file_size
        sub_df = pd.read_csv(f)
        df = pd.concat([df, sub_df])

    dict['data_size'] = size / (1024 * 1024 * 1024)

    start_time = time.time()
    sorted_series = df['close'].sort_values()
    end_time = time.time()
    print(f"Time taken to sort data --- {end_time - start_time} seconds ---")
    dict['sort_exec_time'] = end_time - start_time

    start_time = time.time()
    sorted_series = df['close'].sum()
    end_time = time.time()
    print(f"Time taken to sum data --- {end_time - start_time} seconds ---")
    dict['sum_exec_time'] = end_time - start_time

    row_list.append(dict)

# print(row_list)

pandas_stats = pd.DataFrame(row_list, columns=['data_size', 'sort_exec_time', 'sum_exec_time'])
print(pandas_stats.head())
pandas_stats.to_csv('sort_data.csv')

# Plot the comparison of data gathered for all the benchmarks
plt.figure(1)

# plot sort data stats
plt.subplot(2, 1, 1)
plt.plot(pandas_stats['data_size'], pandas_stats['sort_exec_time'], color='red', label='pandas')
plt.xlabel('Data size(GB)')
plt.ylabel('Time taken(seconds)')
plt.legend()
plt.title('Time taken to sort data in pandas')

# plot sum data stats
plt.subplot(2, 1, 2)
plt.plot(pandas_stats['data_size'], pandas_stats['sum_exec_time'], color='red', label='pandas')
plt.xlabel('Data size(GB)')
plt.ylabel('Time taken(seconds)')
plt.legend()
plt.title('Time taken to sum data in pandas')

# show the plot
plt.subplots_adjust(hspace=0.5)
plt.show()
