import pandas as pd
from influxdb_client import InfluxDBClient #type: ignore

class DataLoading:
    
    
    def __init__(self) -> None:
        pass
        
    
    def load_data(self, file_path) -> pd.DataFrame:
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            print('Exception was occured during the data loading', e)
            
        
    def load_data_from_influx(self, token, org, url, bucket):
        try:
            client = InfluxDBClient(url=url, token=token, org=org)
            query_api = client.query_api()
            
            print('connected to the influxdb.')
            print(f'Fetching the data from database : {bucket}')
            
            flux_query = f"""
            from(bucket: "{bucket}")
            |> range(start: 2023-01-01T00:00:00Z)  
            |> filter(fn: (r) => r._measurement == "sensor_readings")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            """
            tables = query_api.query(flux_query)
            data = []
            for table in tables:
                for record in table.records:
                    data.append(record.values)

            df = pd.DataFrame(data)
            df.drop(columns=['result', 'table', '_start', '_stop', '_time', '_measurement'], axis=1, inplace=True)
            print('data has been loaded...')
            client.close()
            return df
            
        except Exception as e:
            print(f"Error in data loading from influxdb: {str(e)}")

        
        
        
        