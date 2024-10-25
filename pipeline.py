from data_loading import DataLoading
from data_cleaning import DataCleaning
from clustering import perform_knn_classification, plot_2d_pca_with_clusters, perform_pca_2d
from sklearn.metrics import classification_report
import logging

class MLOpsPipeline:
    def __init__(self) -> None:
        self.token = "UXHZ0-LEw4O9wBEVm_TpODVBDit6x86fQmj87y5Fuq7Mgaz0vrZhG8ponUap5DUshmWfFfiG8pQBBeAJ4IZIdQ=="
        self.org = "IISc"
        self.url = "http://13.52.83.241:8086"
        self.bucket = "sensor_data"
        self.file_path = r"C:\Users\admin\OneDrive - Talentpace\15_site_data\telemetry\Site-2791514.csv"

        # Initialize components
        self.data_loader = DataLoading()
        self.data_cleaner = DataCleaning()

        # Set up logging
        logging.basicConfig(level=logging.INFO)

    def run(self):
        # Step 1: Load Data
        logging.info("Loading data...")
        data = self.data_loader.load_data_from_influx(self.token, self.org, self.url, self.bucket)
        
        # Step 2: Clean Data
        logging.info("Cleaning data...")
        data = self.data_cleaner.data_cleaning(data)
        data = data.head(20000)  # Limit data for testing
        logging.info(f"Cleaned data shape: {data.shape}")
        
        # Step 3: Create Labels
        logging.info("Creating labels...")
        data = self.data_cleaner.creating_labels(data)
        labels = data['label']
        
        # Step 4: Standardize Data
        logging.info("Standardizing data...")
        data = self.data_cleaner.standardize_data(data)

        # Step 5: Perform PCA
        logging.info("Performing PCA...")
        principal_components = perform_pca_2d(data)

        # Step 6: Train Model
        logging.info("Training KNN classifier...")
        cluster_labels = perform_knn_classification(principal_components, labels, n_neighbors=3)

        # Step 7: Evaluate Model
        logging.info("Evaluating model...")
        print(classification_report(labels, cluster_labels))

        # Step 8: Plot Results
        logging.info("Plotting results...")
        plot_2d_pca_with_clusters(principal_components, cluster_labels)
        
        logging.info("Pipeline execution completed.")

if __name__ == '__main__':
    pipeline = MLOpsPipeline()
    pipeline.run()
