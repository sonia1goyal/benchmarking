# benchmarking
Repository for spark vs pandas benchmarking

4 benchmarks are performed for comparing pandas vs spark.
1. Time taken to load data in dataframe
2. SQL 1: select max(close) from data
3. SQL 2: select count(distinct volume) from data
4. SQL 3: select sum(high) from data group by date

2 benchmarks performed on pandas
1. Sort data of a column for different data sizes
2. Find sum of a column for different data sizes
