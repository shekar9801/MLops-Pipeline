import mlflow  # type: ignore
import mlflow.sklearn  # type: ignore
from data_loading import DataLoading
from data_cleaning import DataCleaning
from clustering import perform_knn_classification, plot_2d_pca_with_clusters, perform_pca_2d
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import logging
import time
import psutil  # To log system metrics

class MLOpsPipeline:
    def __init__(self, n_neighbors=3) -> None:
        self.token = "UXHZ0-LEw4O9wBEVm_TpODVBDit6x86fQmj87y5Fuq7Mgaz0vrZhG8ponUap5DUshmWfFfiG8pQBBeAJ4IZIdQ=="
        self.org = "IISc"
        self.url = "http://13.52.83.241:8086"
        self.bucket = "sensor_data"
        self.file_path = r"C:\Users\admin\OneDrive - Talentpace\15_site_data\telemetry\Site-2791514.csv"
        self.n_neighbors = n_neighbors

        # Initialize components
        self.data_loader = DataLoading()
        self.data_cleaner = DataCleaning()

        # Set up logging
        logging.basicConfig(level=logging.INFO)

        # Set the MLflow tracking URI
        mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Your local MLflow server URI

        # Set the experiment name
        mlflow.set_experiment("knn clustering")  # Your specified experiment name

    def run(self):
        # Start an MLflow run
        start_time = time.time()  # Start timer
        with mlflow.start_run() as run:
            # Step 1: Load Data
            logging.info("Loading data...")
            data = self.data_loader.load_data_from_influx(self.token, self.org, self.url, self.bucket)

            # Step 2: Clean Data
            logging.info("Cleaning data...")
            data = self.data_cleaner.data_cleaning(data)
            data = data.head(10000)  
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
            knn_model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            knn_model.fit(principal_components, labels)
            cluster_labels = knn_model.predict(principal_components)

            # Step 7: Evaluate Model
            logging.info("Evaluating model...")
            report = classification_report(labels, cluster_labels, output_dict=True)
            print(report)

            # Log metrics and parameters to MLflow
            mlflow.log_param("model_type", "KNN")
            mlflow.log_param("n_neighbors", self.n_neighbors)
            mlflow.log_metric("accuracy", report['accuracy'])

            # Log additional metadata
            mlflow.log_param("data_shape", data.shape)
            mlflow.log_param("features_used", list(data.columns))
            mlflow.log_param("Database", 'InfluxDB')
            mlflow.log_param("dataset", 'Enphase')
            mlflow.log_param("dataset_description", "Telemetry data from Site 2791514, includes various sensor metrics.")
            mlflow.set_tags({"project": "KNN Clustering", "environment": "production"})

            # Log system metrics
            mlflow.log_param("cpu_usage_percent", psutil.cpu_percent())
            mlflow.log_param("memory_usage_percent", psutil.virtual_memory().percent)

            # Log the trained KNN model with an input example
            input_example = principal_components[:5] 
            mlflow.sklearn.log_model(knn_model, "knn_model", input_example=input_example)

            # Register the model in MLflow Model Registry
            mlflow.register_model(f"runs:/{run.info.run_id}/knn_model", "KNNModel")

            # Step 8: Plot Results
            logging.info("Plotting results...")
            plot_2d_pca_with_clusters(principal_components, cluster_labels)

            # Calculate duration
            duration = time.time() - start_time
            mlflow.log_param("execution_duration_seconds", duration)
            logging.info(f"Pipeline execution completed in {duration:.2f} seconds.")

if __name__ == '__main__':
    pipeline = MLOpsPipeline(n_neighbors=3)  
    pipeline.run()
