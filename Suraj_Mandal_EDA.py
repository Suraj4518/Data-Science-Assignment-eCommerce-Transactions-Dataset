import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Perform EDA
def perform_eda(customers, products, transactions):
    print("Customers Data Overview:")
    print(customers.info())
    print(customers.describe())
    
    print("\nProducts Data Overview:")
    print(products.info())
    print(products.describe())
    
    print("\nTransactions Data Overview:")
    print(transactions.info())
    print(transactions.describe())

    # Example visualization
    sns.countplot(x='Region', data=customers)
    plt.title("Customer Count by Region")
    plt.show()

if __name__ == "__main__":
    from load_data import load_data
    
    # Load datasets
    customers, products, transactions = load_data()
    
    # Perform EDA
    perform_eda(customers, products, transactions)
