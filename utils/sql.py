# LOAD DATASET FROM DATABASE
load_dataset_sql = ("""
                    SELECT passenger_count,
                            trip_distance,
                            rate_code_des,
                            pmt_type_des,
                            pu_hour,
                            do_hour,
                            travel_day,
                            fare_amount
                    FROM greentaxi
                    WHERE total_amount > 0 And trip_distance > 0 And lpep_dropoff_datetime between '2023-01-01' And '2023-01-31'
                    """)