from data_loading import DataLoading
from data_cleaning import DataCleaning
from clustering import perform_knn_classification, plot_2d_pca_with_clusters, perform_pca_2d


class Main:
    def __init__(self) -> None:
        self.token = "UXHZ0-LEw4O9wBEVm_TpODVBDit6x86fQmj87y5Fuq7Mgaz0vrZhG8ponUap5DUshmWfFfiG8pQBBeAJ4IZIdQ=="
        self.org = "IISc"
        self.url = "http://13.52.83.241:8086"
        self.bucket = "sensor_data"
        self.file_path =  r"C:\Users\admin\OneDrive - Talentpace\15_site_data\telemetry\Site-2791514.csv"
        
    
    
if __name__=='__main__':
    main = Main()
    data_loader = DataLoading()
    data_cleaner = DataCleaning()
    
    # data = data_loader.load_data(main.file_path)
    data = data_loader.load_data_from_influx(main.token, main.org, main.url, main.bucket)
    data = data_cleaner.data_cleaning(data)
    data = data.head(10000)
    print(data.head())
    
    data = data_cleaner.creating_labels(data)
    labels = data['label']
    
    data = data_cleaner.standardize_data(data)
    
    print(data.head())
    
    principal_components = perform_pca_2d(data)
    cluster_labels = perform_knn_classification(principal_components, labels, n_neighbors=3)
    plot_2d_pca_with_clusters(principal_components, cluster_labels)
    
    
    
    