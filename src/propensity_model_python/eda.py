import matplotlib.pyplot as plt
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load customer data from a CSV file."""
    return pd.read_csv(path)


def summarize_data(data: pd.DataFrame) -> None:
    """Print a basic exploratory summary of the dataset."""

    print("\n--- SHAPE ---")
    print(data.shape)

    print("\n--- FIRST ROWS ---")
    print(data.head())

    print("\n--- DATA TYPES ---")
    print(data.dtypes)

    print("\n--- MISSING VALUES ---")
    print(data.isna().sum())

    print("\n--- DUPLICATES ---")
    print(data.duplicated().sum())

    print("\n--- DESCRIPTIVE STATISTICS ---")
    print(data.describe())

    print("\n--- TARGET DISTRIBUTION ---")
    print(data["target"].value_counts())

    print("\n--- TARGET DISTRIBUTION (%) ---")
    print(data["target"].value_counts(normalize=True))

    print("\n--- MEANS BY TARGET ---")
    print(
        data.groupby("target")[
            [
                "age",
                "tenure_months",
                "monthly_spend",
                "previous_purchases",
                "campaign_contacts",
                "digital_customer",
            ]
        ].mean()
    )
    
def plot_target_distribution(data: pd.DataFrame) -> None:
    """Plot the distribution of the target variable."""

    counts = data["target"].value_counts().sort_index()

    counts.plot(kind="bar")

    plt.title("Target distribution")
    plt.xlabel("Target")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig("outputs/target_distribution.png")
    plt.close()
    
def plot_monthly_spend_distribution(data: pd.DataFrame) -> None:
    """Plot the distribution of monthly spending."""

    data["monthly_spend"].plot(
        kind="hist",
        bins=30,
    )

    plt.title("Monthly spend distribution")
    plt.xlabel("Monthly spend")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("outputs/monthly_spend_distribution.png")
    plt.close()

def plot_spend_by_target(data: pd.DataFrame) -> None:
    """Plot average monthly spend by target."""

    means = (
        data.groupby("target")["monthly_spend"]
        .mean()
    )

    means.plot(kind="bar")

    plt.title("Average monthly spend by target")
    plt.xlabel("Target")
    plt.ylabel("Average monthly spend")
    plt.tight_layout()
    plt.savefig("outputs/spend_by_target.png")


if __name__ == "__main__":
    data = load_data("data/raw/customers.csv")

    summarize_data(data)

    plot_target_distribution(data)
    plot_monthly_spend_distribution(data)
    plot_spend_by_target(data)