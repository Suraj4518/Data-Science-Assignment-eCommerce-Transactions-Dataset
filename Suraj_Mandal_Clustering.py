import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
import seaborn as sns
import matplotlib.pyplot as plt
from generate_pdf import generate_pdf

def prepare_data(customers, transactions):
    # Check for required columns
    required_columns = ['TotalValue', 'Quantity']
    for column in required_columns:
        if column not in transactions.columns:
            raise ValueError(f"Missing required column: {column}")

    # Aggregate transaction data by CustomerID
    transaction_summary = transactions.groupby('CustomerID').agg({
        'TotalValue': 'sum',
        'Quantity': 'sum'
    }).reset_index()

    # Merge with customer profiles
    customer_data = customers.merge(transaction_summary, on='CustomerID', how='left').fillna(0)
    return customer_data

def perform_clustering(customer_data, n_clusters=3):
    # Select relevant features
    features = customer_data[['TotalValue', 'Quantity']]

    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customer_data['Cluster'] = kmeans.fit_predict(scaled_features)

    return customer_data, kmeans, scaled_features

def evaluate_clustering(scaled_features, clusters):
    # Calculate Davies-Bouldin Index
    db_index = davies_bouldin_score(scaled_features, clusters)
    return db_index

def visualize_clusters(customer_data):
    # Visualize clusters using a scatter plot
    sns.scatterplot(data=customer_data, x='TotalValue', y='Quantity', hue='Cluster', palette='viridis')
    plt.title("Customer Segments")
    plt.xlabel("Total Transaction Value")
    plt.ylabel("Total Quantity Purchased")
    plt.show()

def save_results(customer_data, output_file):
    # Prepare insights for PDF
    insights = []
    for index, row in customer_data.iterrows():
        insights.append(f"Customer ID: {row['CustomerID']}, Cluster: {row['Cluster']}, Total Value: {row['TotalValue']}, Quantity: {row['Quantity']}")
    
    # Generate PDF report
    generate_pdf(insights, output_file)

if __name__ == "__main__":
    from load_data import load_data

    # Load data
    customers, _, transactions = load_data()

    # Prepare data
    customer_data = prepare_data(customers, transactions)

    # Perform clustering (try different cluster numbers)
    customer_data, kmeans, scaled_features = perform_clustering(customer_data, n_clusters=3)

    # Evaluate clustering performance
    db_index = evaluate_clustering(scaled_features, customer_data['Cluster'])
    print(f"Davies-Bouldin Index: {db_index}")

    # Visualize clusters
    visualize_clusters(customer_data)

    # Save clustering results to PDF
    save_results(customer_data, "../output/Suraj_Mandal_Clustering.pdf")
