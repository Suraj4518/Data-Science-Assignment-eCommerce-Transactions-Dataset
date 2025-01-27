import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import os

def prepare_data(customers, transactions):
    # Aggregate transaction data by CustomerID
    transaction_summary = transactions.groupby('CustomerID').agg({
        'TotalValue': 'sum',
        'Quantity': 'sum'
    }).reset_index()

    # Merge with customer profiles
    customer_data = customers.merge(transaction_summary, on='CustomerID', how='left').fillna(0)
    return customer_data

def calculate_similarity(customer_data):
    # Normalize numerical features for similarity calculation
    scaler = MinMaxScaler()
    numerical_data = scaler.fit_transform(customer_data[['TotalValue', 'Quantity']])

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(numerical_data)
    return similarity_matrix

def find_lookalikes(customer_data, similarity_matrix, customer_ids):
    # Prepare the lookalike dictionary
    lookalikes = {}
    for idx, customer_id in enumerate(customer_ids):
        # Get similarity scores for the given customer
        similarities = similarity_matrix[idx]
        # Sort by similarity score (excluding the customer itself)
        similar_customers = sorted(
            [(customer_data.iloc[i]['CustomerID'], score) for i, score in enumerate(similarities) if i != idx],
            key=lambda x: x[1],
            reverse=True
        )
        # Take the top 3 similar customers
        lookalikes[customer_id] = similar_customers[:3]
    return lookalikes

def save_lookalikes(lookalikes, output_file):
    # Convert lookalikes dictionary to a DataFrame for saving
    rows = []
    for customer_id, similar_customers in lookalikes.items():
        for similar_customer_id, score in similar_customers:
            rows.append([customer_id, similar_customer_id, score])
    
    lookalikes_df = pd.DataFrame(rows, columns=['CustomerID', 'SimilarCustomerID', 'SimilarityScore'])
    # Update the output file path
    output_file = os.path.join(os.path.dirname(__file__), '../output/Suraj_Mandal_Lookalike.csv')
    lookalikes_df.to_csv(output_file, index=False)
    print(f"Lookalike data saved to: {output_file}")

if __name__ == "__main__":
    from load_data import load_data

    # Load data
    customers, _, transactions = load_data()

    # Prepare data
    customer_data = prepare_data(customers, transactions)

    # Calculate similarity
    similarity_matrix = calculate_similarity(customer_data)

    # Find lookalikes for the first 20 customers
    customer_ids = customer_data['CustomerID'][:20]
    lookalikes = find_lookalikes(customer_data, similarity_matrix, customer_ids)

    # Save lookalikes to CSV
    save_lookalikes(lookalikes, "../output/Suraj_Mandal_Lookalike.csv")
