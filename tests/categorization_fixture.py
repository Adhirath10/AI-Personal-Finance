"""A transaction table shaped like the project dataset ("<Merchant> Transaction")."""
import sqlite3

import numpy as np

MERCHANTS = {
    "Food": ["Swiggy", "Zomato", "Restaurant", "Cafe", "Food Delivery"],
    "Groceries": ["BigBasket", "Blinkit", "DMart", "Reliance Smart", "Local Grocery Store"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "Ajio"],
    "Transport": ["Uber", "Ola", "Metro", "Auto", "Fuel Station"],
    "Travel": ["IRCTC", "MakeMyTrip", "Flight", "Hotel", "Bus Booking"],
    "Bills": ["Electricity Bill", "Water Bill", "Gas Bill", "Internet Bill", "Mobile Bill"],
    "Subscription": ["Netflix", "Spotify", "Amazon Prime", "YouTube Premium", "Cloud Storage"],
    "Entertainment": ["Movie", "Concert", "Gaming", "OTT"],
    "Healthcare": ["Pharmacy", "Hospital", "Clinic", "Diagnostic Lab"],
    "Education": ["Udemy", "Coursera", "Books", "College Fee", "Course Fee"],
    "Investment": ["Mutual Fund", "SIP", "FD", "Stocks"],
    "Personal Care": ["Salon", "Gym", "Cosmetics", "Personal Care"],
    "Rent": ["House Rent", "Monthly Rent"],
    "Salary": ["Employer"],
    "Other Income": ["Freelance Client"],
    "Other": ["Miscellaneous", "Gift", "Donation", "Other Expense"],
}


def write_categorization_db(path, rows_per_merchant=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for category, merchants in MERCHANTS.items():
        for merchant in merchants:
            for index in range(rows_per_merchant):
                description = f"{merchant} Transaction" if rng.random() < 0.8 else f"Payment {merchant}"
                if category == "Salary":
                    description = "Monthly Salary Credit"
                rows.append((f"T{len(rows)}", description, merchant, category))
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE transactions (transaction_id TEXT, description TEXT, merchant TEXT, category TEXT)")
    connection.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()
    return path
