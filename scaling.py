"""
Data loading time comparison
Amount of memory taken by the process
"""

import pandas as pd
import glob
import os
import time
import psutil
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import argparse


# Add argument for path
parser = argparse.ArgumentParser()
parser.add_argument('path',  help='Path oto the csv file folder')

args = parser.parse_args()
path = args.path
all_files = glob.glob(os.path.join(path, "*.csv"))


# Benchmarking data for pandas
row_list = []
for n in [3, 5, 10, 15, 20, 25, 30, 33]:
    # In my case each csv file is approximately 630 MB so each file s 0.615 GB
    # Dividing by 0.615 will give the number of files required to make n GB
    n = round(n / 0.615)
    print("Number of files to process: ", n)

    start_time = time.time()

    size = 0
    dict = {}
    df = pd.DataFrame()
    for f in all_files[:n]:
        file_size = os.path.getsize(f)
        size = size + file_size
        sub_df = pd.read_csv(f)
        df = pd.concat([df, sub_df])

    dict['data_size'] = size / (1024 * 1024 * 1024)

    process = psutil.Process(os.getpid())
    dict['memory_used'] = process.memory_info().rss/(1024*1024*1024)

    end_time = time.time()
    dict['exec_time'] = end_time - start_time
    dict['number_of_rows'] = len(df.index)

    row_list.append(dict)


# Benchmarking data for Spark
row_list2 = []
spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
for n in [3, 5, 10, 15, 20, 25, 30, 33]:
    # In my case each csv file is approximately 630 MB so each file s 0.615 GB
    # Dividing by 0.615 will give the number of files required to make n GB
    n = round(n / 0.615)
    print("Number of files to process: ", n)

    start_time = time.time()

    dict = {}
    df = spark.read.csv(all_files[0])
    size = os.path.getsize(all_files[0])
    for f in all_files[1:n]:
        file_size = os.path.getsize(f)
        size = size + file_size
        sub_df = spark.read.csv(f)
        df = df.union(sub_df)

    dict['data_size'] = size / (1024 * 1024 * 1024)

    process = psutil.Process(os.getpid())
    dict['memory_used'] = process.memory_info().rss/(1024*1024*1024)

    end_time = time.time()
    dict['exec_time'] = end_time - start_time
    dict['number_of_rows'] = df.count()

    row_list2.append(dict)


# Load the stats in a pandas dataframe for easy usage
pandas_stats = pd.DataFrame(row_list, columns=['data_size', 'memory_used', 'exec_time', 'number_of_rows'])
print(pandas_stats.head())

spark_stats = pd.DataFrame(row_list2, columns=['data_size', 'memory_used', 'exec_time', 'number_of_rows'])
print(spark_stats.head())


# Plot the comparison of data gathered for all the benchmarks
plt.plot(pandas_stats['data_size'], pandas_stats['exec_time'], color='red', label='pandas')
plt.plot(spark_stats['data_size'], spark_stats['exec_time'], color='blue', label='pyspark')
plt.xlabel('Data size(GB)')
plt.ylabel('Time taken(seconds)')
plt.legend()
plt.title('Time taken to load data, pandas vs pyspark')

# show the plot
plt.show()
