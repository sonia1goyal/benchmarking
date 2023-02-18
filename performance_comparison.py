"""
Benchmarking of Pandas vs Spark performance
Perform below benchmarks on it.
SQL 1: select max(close) from data
SQL 2: select count(distinct volume) from data
SQL 3: select sum(high) from data group by date
"""

import pandas as pd
import glob
import os
import time
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('path',  help='Path oto the csv file folder')
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
    max_val = df['close'].max()
    end_time = time.time()
    print(f"Time taken to get max --- {end_time - start_time} seconds ---")
    dict['max_query_exec_time'] = end_time - start_time

    start_time = time.time()
    count = df['volume'].nunique()
    end_time = time.time()
    dict['count_query_exec_time'] = end_time - start_time

    start_time = time.time()
    count = df[['date', 'high']].groupby('date').sum()
    end_time = time.time()
    dict['group_query_exec_time'] = end_time - start_time

    row_list.append(dict)

# Benchmarking data for Spark
row_list2 = []
spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
for n in [2, 4, 6, 8, 10, 15, 20]:
    n = round(n / 0.615)
    print("Number of files to process: ", n)

    dict = {}
    df = spark.read.csv(all_files[0], sep=',', multiLine=True, header=True)
    size = os.path.getsize(all_files[0])
    for f in all_files[1:n]:
        file_size = os.path.getsize(f)
        size = size + file_size
        sub_df = spark.read.csv(f,  sep=',', multiLine=True, header=True)
        df = df.union(sub_df)

    df.createOrReplaceTempView("temp")
    dict['data_size'] = size / (1024 * 1024 * 1024)

    start_time = time.time()
    max_value = spark.sql("select max(close) from temp").collect()
    # max_value = df.agg({'close': 'max'}).collect()
    end_time = time.time()
    print(f"Time taken--- {end_time - start_time} seconds ---")
    dict['max_query_exec_time'] = end_time - start_time

    start_time = time.time()
    count = spark.sql("select count(distinct volume) from temp").collect()
    end_time = time.time()
    dict['count_query_exec_time'] = end_time - start_time

    start_time = time.time()
    max_val = spark.sql("select sum(high) from temp group by date").collect()
    end_time = time.time()
    dict['group_query_exec_time'] = end_time - start_time

    row_list2.append(dict)



# Load the stats in a pandas dataframe for easy usage
pandas_stats = pd.DataFrame(row_list, columns=['data_size', 'max_query_exec_time', 'count_query_exec_time',
                                               'group_query_exec_time'])
print(pandas_stats.head())

spark_stats = pd.DataFrame(row_list2, columns=['data_size', 'max_query_exec_time', 'count_query_exec_time',
                                               'group_query_exec_time'])
print(spark_stats.head())


# Plot the comparison of data gathered for all the benchmarks
# SQL 1
plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(pandas_stats['data_size'], pandas_stats['max_query_exec_time'], color='red', label='pandas')
plt.plot(spark_stats['data_size'], spark_stats['max_query_exec_time'], color='blue', label='pyspark')
plt.xlabel('Data size(GB)')
plt.ylabel('Time taken(seconds)')
plt.legend()
plt.title('Time taken by max query, pandas vs pyspark')

# SQL 2
plt.subplot(3, 1, 2)
plt.plot(pandas_stats['data_size'], pandas_stats['count_query_exec_time'], color='red', label='pandas')
plt.plot(spark_stats['data_size'], spark_stats['count_query_exec_time'], color='blue', label='pyspark')
plt.xlabel('Data size(GB)')
plt.ylabel('Time taken(seconds)')
plt.legend()
plt.title('Time taken by count distinct query, pandas vs pyspark')

# SQL 3
plt.subplot(3, 1, 3)
plt.plot(pandas_stats['data_size'], pandas_stats['group_query_exec_time'], color='red', label='pandas')
plt.plot(spark_stats['data_size'], spark_stats['group_query_exec_time'], color='blue', label='pyspark')
plt.xlabel('Data size(GB)')
plt.ylabel('Time taken(seconds)')
plt.legend()
plt.title('Time taken by group by sum query, pandas vs pyspark')

# show the plot
plt.subplots_adjust(hspace=0.5)
plt.show()
