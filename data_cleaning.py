from data_loading import DataLoading
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class DataCleaning:
    
    def __init__(self) -> None:
        pass
    
    def data_cleaning(self, data=None, use_columns=None):
        try:
            if data is None:
                raise ValueError('The dataset is empty...')
            
            if use_columns is None:
                # print('The column list is empty...')
                use_columns = data.columns
            
            if data is not None and use_columns is not None:    
                data = data[use_columns]
                data.dropna(inplace=True)
                return data
            else:
                raise ValueError('Something went wrong...')

        except Exception as e:
            print(f"Error in data_cleaning: {str(e)}")
            
            
    def creating_labels(self, data):
        
        if not {'ac_voltage', 'energy_produced'}.issubset(data.columns):
            raise ValueError("DataFrame must contain 'ac_voltage' and 'energy_produced' columns.")

        data['label'] = data.apply(lambda x: 1 if (x['ac_voltage'] < 225 or x['ac_voltage'] > 260) and x['energy_produced'] == 0 else 0, axis=1)
        
        return data

    
    def standardize_data(self, data):
        try:
            # use_columns = [column for column in data.columns if column not in ['label']]
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            scaled_data = pd.DataFrame(data = scaled_data, columns=data.columns)
            return scaled_data
        
        except Exception as e:
            print(f"Error in standardize_data: {str(e)}")
            
    