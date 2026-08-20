import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/processed/verification_dataset.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET OVERVIEW ==========\n")

    print("Dataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\n========== TARGET DISTRIBUTION ==========\n")

    print(df["needs_human_review"].value_counts())

    print("\n========== CATEGORY DISTRIBUTION ==========\n")

    print(df["category"].value_counts())

    print("\n========== AVERAGE FEATURES BY RISK ==========\n")

    print(
        df.groupby("needs_human_review")[
            [
                "question_length",
                "response_length",
                "word_count",
                "uncertainty_count",
            ]
        ].mean()
    )

    # -----------------------------
    # Risk distribution
    # -----------------------------

    df["needs_human_review"].value_counts().sort_index().plot(
        kind="bar"
    )

    plt.title("Human Review Requirement")
    plt.xlabel("Needs Human Review")
    plt.ylabel("Number of Responses")
    plt.xticks(
        [0, 1],
        ["No", "Yes"],
        rotation=0
    )

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Risk by category
    # -----------------------------

    category_risk = pd.crosstab(
        df["category"],
        df["needs_human_review"]
    )

    category_risk.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Human Review Requirement by Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Responses")
    plt.legend(
        ["No Human Review", "Human Review"]
    )

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Response length vs risk
    # -----------------------------

    df.boxplot(
        column="response_length",
        by="needs_human_review"
    )

    plt.title("Response Length vs Human Review")
    plt.suptitle("")
    plt.xlabel("Needs Human Review")
    plt.ylabel("Response Length")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()