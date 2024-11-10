-- Setup records 
create schema tpch_benchmark;
CREATE TABLE benchmark_results (
    query_id STRING,
    tpch_query integer, 
    dataset text,
    warehouse_size text,
    repetition_num integer,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    elapsed_time_ms NUMBER,
    query_text text
);
--show tables history like 'benchmark_results';
--undrop table benchmark_results;
--alter table benchmark_results rename to temp;
--select * from benchmark_results;
--ALTER SESSION SET USE_CACHED_RESULT=FALSE;

CREATE OR REPLACE WAREHOUSE chipmunk_wh_xs
  WAREHOUSE_SIZE = 'XSMALL';
CREATE OR REPLACE WAREHOUSE chipmunk_wh_s
    WAREHOUSE_SIZE = 'SMALL';
CREATE OR REPLACE WAREHOUSE chipmunk_wh_m
    WAREHOUSE_SIZE = 'MEDIUM';
CREATE OR REPLACE WAREHOUSE chipmunk_wh_l
    WAREHOUSE_SIZE = 'LARGE';
use warehouse chipmunk_wh_xs;
use warehouse chipmunk_wh_s;
use warehouse chipmunk_wh_m;
use warehouse chipmunk_wh_l;


-- TPC-h query 1 
--(source:https://docs.snowflake.com/en/user-guide/sample-data-tpch)
set dataset= 'tpch_sf1000';
set warehouse_size = (SELECT CURRENT_WAREHOUSE());
set repetition_num = 1;
-- TPC-H query 1.
select
       l_returnflag,
       l_linestatus,
       sum(l_quantity) as sum_qty,
       sum(l_extendedprice) as sum_base_price,
       sum(l_extendedprice * (1-l_discount)) as sum_disc_price,
       sum(l_extendedprice * (1-l_discount) * (1+l_tax)) as sum_charge,
       avg(l_quantity) as avg_qty,
       avg(l_extendedprice) as avg_price,
       avg(l_discount) as avg_disc,
       count(*) as count_order
 from
       SNOWFLAKE_SAMPLE_DATA.tpch_sf1000.lineitem
 where
       l_shipdate <= dateadd(day, -90, to_date('1998-12-01'))
 group by
       l_returnflag,
       l_linestatus
 order by
       l_returnflag,
       l_linestatus;
// Get results
set query_id = LAST_QUERY_ID();
set tpch_query = 1;
set start_time = (SELECT start_time FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())  where query_id = $query_id ORDER BY END_TIME DESC LIMIT 1);
set end_time = (SELECT end_time FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY()) where query_id = $query_id ORDER BY END_TIME DESC LIMIT 1);
set query_text = (SELECT left(query_text, 256) FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())  where query_id = $query_id  ORDER BY END_TIME DESC LIMIT 1);
set total_elapsed_time = (SELECT total_elapsed_time FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY()) where query_id = $query_id   ORDER BY END_TIME DESC LIMIT 1);

INSERT INTO benchmark_results (query_id, tpch_query,dataset, warehouse_size, repetition_num, start_time, end_time, elapsed_time_ms, query_text)
VALUES (LAST_QUERY_ID() ,$tpch_query, $dataset, $warehouse_size, $repetition_num, $start_time, $end_time, $total_elapsed_time, $query_text);

-- TPC-H query 5.
-- https://github.com/apache/impala/blob/master/testdata/workloads/tpch/queries/tpch-q5.test
select
  n_name,
  sum(l_extendedprice * (1 - l_discount)) as revenue
from
  SNOWFLAKE_SAMPLE_DATA.tpch_sf1000.customer,
  SNOWFLAKE_SAMPLE_DATA.tpch_sf1000.orders,
  SNOWFLAKE_SAMPLE_DATA.tpch_sf1000.lineitem,
  SNOWFLAKE_SAMPLE_DATA.tpch_sf1000.supplier,
  SNOWFLAKE_SAMPLE_DATA.tpch_sf1000.nation,
  SNOWFLAKE_SAMPLE_DATA.tpch_sf1000.region
where
  c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and l_suppkey = s_suppkey
  and c_nationkey = s_nationkey
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'ASIA'
  and o_orderdate >= '1994-01-01'
  and o_orderdate < '1995-01-01'
group by
  n_name
order by
  revenue desc;

set query_id = LAST_QUERY_ID();
set tpch_query = 5;
set start_time = (SELECT start_time FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())  where query_id = $query_id ORDER BY END_TIME DESC LIMIT 1);
set end_time = (SELECT end_time FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY()) where query_id = $query_id ORDER BY END_TIME DESC LIMIT 1);
set query_text = (SELECT left(query_text, 256) FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())  where query_id = $query_id  ORDER BY END_TIME DESC LIMIT 1);
set total_elapsed_time = (SELECT total_elapsed_time FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY()) where query_id = $query_id   ORDER BY END_TIME DESC LIMIT 1);

INSERT INTO benchmark_results (query_id, tpch_query,dataset, warehouse_size, repetition_num, start_time, end_time, elapsed_time_ms, query_text)
VALUES (LAST_QUERY_ID() ,$tpch_query, $dataset, $warehouse_size, $repetition_num, $start_time, $end_time, $total_elapsed_time, $query_text);

-- 18 TPC-H/TPC-R Large Volume Customer Query (Q18)
-- Source: https://github.com/apache/impala/blob/master/testdata/workloads/tpch/queries/tpch-q18.test
select
  c_name,
  c_custkey,
  o_orderkey,
  o_orderdate,
  o_totalprice,
  sum(l_quantity)
from
  SNOWFLAKE_SAMPLE_DATA.tpch_sf1000.customer,
  SNOWFLAKE_SAMPLE_DATA.tpch_sf1000.orders,
  SNOWFLAKE_SAMPLE_DATA.tpch_sf1000.lineitem
where
  o_orderkey in (
    select
      l_orderkey
    from
      SNOWFLAKE_SAMPLE_DATA.tpch_sf1000.lineitem
    group by
      l_orderkey
    having
      sum(l_quantity) > 300
    )
  and c_custkey = o_custkey
  and o_orderkey = l_orderkey
group by
  c_name,
  c_custkey,
  o_orderkey,
  o_orderdate,
  o_totalprice
order by
  o_totalprice desc,
  o_orderdate,
  o_orderkey
limit 100;

set query_id = LAST_QUERY_ID();
set tpch_query = 18;
set start_time = (SELECT start_time FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())  where query_id = $query_id ORDER BY END_TIME DESC LIMIT 1);
set end_time = (SELECT end_time FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY()) where query_id = $query_id ORDER BY END_TIME DESC LIMIT 1);
set query_text = (SELECT left(query_text, 256) FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())  where query_id = $query_id  ORDER BY END_TIME DESC LIMIT 1);
set total_elapsed_time = (SELECT total_elapsed_time FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY()) where query_id = $query_id   ORDER BY END_TIME DESC LIMIT 1);

INSERT INTO benchmark_results (query_id, tpch_query,dataset, warehouse_size, repetition_num, start_time, end_time, elapsed_time_ms, query_text)
VALUES (LAST_QUERY_ID() ,$tpch_query, $dataset, $warehouse_size, $repetition_num, $start_time, $end_time, $total_elapsed_time, $query_text);

select * from benchmark_results order by end_time desc;