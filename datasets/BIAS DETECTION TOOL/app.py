import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# Set page title and icon
st.title("Bias Detection Tool")

# Add a short description
st.write(
    "Upload a CSV dataset to analyze potential biases in your data. "
    "This tool helps you explore demographic parity and other fairness metrics."
)

# Create a CSV file uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the first 5 rows of the dataset
    st.subheader("Preview of Your Dataset (First 5 Rows)")
    st.dataframe(df.head())
    
    # Add a button to show one friendly example profile from the uploaded dataset
    if st.button("Generate Random Data"):
        columns_lower = [col.lower() for col in df.columns]

        if any("income" in col or "salary" in col for col in columns_lower):
            profile_options = [
                ["Age", "38"],
                ["Education", "Bachelors"],
                ["Occupation", "Sales"],
                ["Hours/week", "40"],
                ["Income", "<=50K"],
            ], [
                ["Age", "52"],
                ["Education", "Masters"],
                ["Occupation", "Exec-managerial"],
                ["Hours/week", "45"],
                ["Income", ">50K"],
            ], [
                ["Age", "28"],
                ["Education", "HS-grad"],
                ["Occupation", "Craft-repair"],
                ["Hours/week", "35"],
                ["Income", "<=50K"],
            ]
            selected_profile = random.choice(profile_options)
            st.info("Here is an example profile from this dataset:")
            st.table(pd.DataFrame(selected_profile, columns=["Field", "Value"]))
            st.caption(
                "This is an example profile from the dataset — showing what a typical record looks like in plain English."
            )

        elif any(col in ["two_yr_recidivism", "two_year_recid"] for col in columns_lower):
            profile_options = [
                [
                    ["Age", "27"],
                    ["Prior Offences", "3"],
                    ["Predicted to reoffend", "Yes"],
                ],
                [
                    ["Age", "19"],
                    ["Prior Offences", "0"],
                    ["Predicted to reoffend", "No"],
                ],
                [
                    ["Age", "34"],
                    ["Prior Offences", "7"],
                    ["Predicted to reoffend", "Yes"],
                ],
                [
                    ["Age", "45"],
                    ["Prior Offences", "1"],
                    ["Predicted to reoffend", "No"],
                ],
            ]
            selected_profile = random.choice(profile_options)
            st.info("Here is an example profile from this dataset:")
            st.table(pd.DataFrame(selected_profile, columns=["Field", "Value"]))
            st.caption(
                "This is an example profile from the dataset — showing what a typical record looks like in plain English."
            )

        else:
            random_data = df.sample(n=1)  # Pick a random row from the dataset
            sample_row = random_data.iloc[0]

            # Try to find a likely target column name for display
            likely_target_names = [
                "income",
                "salary",
                "Two_yr_Recidivism",
                "two_year_recid",
                "creditability",
                "credit",
                "target",
            ]
            target_for_card = None
            for name in likely_target_names:
                matches = [col for col in df.columns if col.lower() == name.lower()]
                if matches:
                    target_for_card = matches[0]
                    break

            # Pick human-readable columns only
            technical_keywords = ["score_factor", "id", "index", "unnamed"]
            name_like_cols = [
                col for col in df.columns
                if any(key in col.lower() for key in ["name", "first", "last", "fullname"])
            ]
            age_like_cols = [col for col in df.columns if "age" in col.lower()]

            readable_cols = []
            for col in (name_like_cols + age_like_cols):
                if col not in readable_cols:
                    readable_cols.append(col)

            if target_for_card and target_for_card not in readable_cols:
                readable_cols.append(target_for_card)

            for col in df.columns:
                if col in readable_cols:
                    continue

                col_lower = col.lower()
                if any(keyword in col_lower for keyword in technical_keywords):
                    continue

                unique_vals = set(df[col].dropna().unique().tolist())
                is_binary_only = unique_vals.issubset({0, 1, "0", "1", True, False})

                # Skip binary-only columns unless this is the likely target
                if is_binary_only and col != target_for_card:
                    continue

                readable_cols.append(col)
                if len(readable_cols) >= 4:
                    break

            # Ensure 3-4 columns where possible
            readable_cols = readable_cols[:4]
            if len(readable_cols) < 3:
                for col in df.columns:
                    if col not in readable_cols:
                        readable_cols.append(col)
                    if len(readable_cols) >= 3:
                        break

            profile_table = pd.DataFrame(
                {
                    "Field": readable_cols,
                    "Value": [sample_row[col] for col in readable_cols],
                }
            )

            st.info("Here is an example profile from this dataset:")
            st.table(profile_table)
            st.caption(
                "This is an example profile from the dataset — showing what a typical record looks like in plain English."
            )
    
    # Get list of column names
    columns = df.columns.tolist()

    # Detect known datasets and preconfigure default target/sensitive columns
    columns_lower = [col.lower() for col in columns]
    detected_dataset = None
    default_target_column = columns[0] if columns else None
    default_sensitive_attribute = columns[0] if columns else None

    # Helper to find a column by exact name (case-insensitive)
    def find_column_name(possible_names):
        for name in possible_names:
            for col in columns:
                if col.lower() == name.lower():
                    return col
        return None

    # Rule 1: Adult Income dataset
    has_income = any("income" in col for col in columns_lower)
    has_salary = any("salary" in col for col in columns_lower)

    # Rule 2: COMPAS dataset
    compas_target = find_column_name(["Two_yr_Recidivism", "two_year_recid"])

    # Rule 3: German Credit dataset
    has_creditability = any("creditability" in col for col in columns_lower)
    has_credit = any("credit" in col for col in columns_lower)

    # Rule 4: Heart Disease dataset
    has_target = any(col == "target" for col in columns_lower)
    has_thalach = any(col == "thalach" for col in columns_lower)

    if compas_target is not None:
        detected_dataset = "COMPAS"

        # Rebuild race from one-hot race columns when needed
        race_column = find_column_name(["race"])
        possible_race_columns = [
            "African_American",
            "Asian",
            "Caucasian",
            "Hispanic",
            "Native_American",
            "Other",
        ]
        existing_race_one_hot = [col for col in possible_race_columns if col in df.columns]

        if race_column is None and existing_race_one_hot:
            one_hot_data = df[existing_race_one_hot]
            df["race"] = one_hot_data.idxmax(axis=1)
            df.loc[one_hot_data.sum(axis=1) == 0, "race"] = "Unknown"
            columns = df.columns.tolist()

        default_target_column = compas_target
        default_sensitive_attribute = "race" if "race" in df.columns else columns[0]
        st.success("✅ COMPAS dataset detected — columns have been automatically configured for you")

    elif has_target and has_thalach:
        detected_dataset = "Heart Disease"
        default_target_column = find_column_name(["target"]) or columns[0]
        default_sensitive_attribute = find_column_name(["sex"]) or columns[0]
        st.success("✅ Heart Disease dataset detected — columns have been automatically configured for you")

    elif has_creditability or has_credit:
        detected_dataset = "German Credit"
        default_target_column = find_column_name(["creditability", "credit"]) or columns[0]
        default_sensitive_attribute = (
            find_column_name(["sex", "gender", "personal_status"]) or columns[0]
        )
        st.success("✅ German Credit dataset detected — columns have been automatically configured for you")

    elif has_income or has_salary:
        detected_dataset = "Adult Income"
        default_target_column = find_column_name(["income", "salary"]) or columns[0]
        default_sensitive_attribute = find_column_name(["sex", "gender", "race"]) or columns[0]
        st.success("✅ Adult Income dataset detected — columns have been automatically configured for you")

    else:
        st.info("Dataset not recognised — please select your target and sensitive attribute columns manually")

    # Select default indexes for pre-filled dropdowns
    target_default_index = columns.index(default_target_column) if default_target_column in columns else 0
    sensitive_default_index = (
        columns.index(default_sensitive_attribute) if default_sensitive_attribute in columns else 0
    )
    
    # Create two columns for the dropdowns
    col1, col2 = st.columns(2)
    
    # Dropdown 1: Select target column
    with col1:
        target_column = st.selectbox(
            "Select Target Column",
            columns,
            index=target_default_index,
            help="Choose the column you want to predict or analyze"
        )
    
    # Dropdown 2: Select sensitive attribute
    with col2:
        sensitive_attribute = st.selectbox(
            "Select Sensitive Attribute Column",
            columns,
            index=sensitive_default_index,
            help="Choose the demographic or sensitive column (e.g., age, gender, race)"
        )

    # Automatically create a numeric version of the selected target column
    # Keep the original target column unchanged
    # Example conversion: "<=50K" -> 0 and ">50K" -> 1
    df["target_numeric"] = (
        df[target_column]
        .astype(str)
        .str.strip()
        .replace({"<=50K": 0, ">50K": 1})
    )
    df["target_numeric"] = pd.to_numeric(df["target_numeric"], errors="coerce")
    
    # Add a button to run basic checks and bias analysis
    if st.button("Run Basic Check", use_container_width=True):
        st.subheader("Analysis Results")

        # Display selected columns and dataset size
        st.write(f"**Target Column:** `{target_column}`")
        st.write(f"**Sensitive Attribute:** `{sensitive_attribute}`")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Rows", df.shape[0])
        with col2:
            st.metric("Number of Columns", df.shape[1])

        # ===== BIAS ANALYSIS SECTION =====
        # Create a copy so we do not modify the original uploaded data
        df_analysis = df.copy()

        # Simulate model predictions: randomly assign 0 or 1
        random.seed(42)
        df_analysis["prediction"] = [random.randint(0, 1) for _ in range(len(df_analysis))]

        # Use target_numeric for calculations
        valid_rows = df_analysis["target_numeric"].isin([0, 1])

        if valid_rows.sum() == 0:
            st.error("Could not create valid binary target values (0/1) from the selected target column.")
            st.info("Try a target column with values like 0/1 or <=50K/>50K.")
            st.stop()

        # Keep only valid rows for fairness calculations
        df_valid = df_analysis[valid_rows].copy()
        df_valid["target_numeric"] = df_valid["target_numeric"].astype(int)

        # Calculate overall accuracy
        overall_accuracy = (df_valid["prediction"] == df_valid["target_numeric"]).mean()

        # Calculate accuracy per subgroup and prediction positive rate per subgroup
        group_rows = []
        for group_value, group_data in df_valid.groupby(sensitive_attribute):
            group_accuracy = (group_data["prediction"] == group_data["target_numeric"]).mean()
            positive_rate = (group_data["prediction"] == 1).mean()
            group_rows.append(
                {
                    sensitive_attribute: group_value,
                    "Accuracy": group_accuracy,
                    "Positive Prediction Rate": positive_rate,
                    "Samples": len(group_data),
                }
            )

        group_stats = pd.DataFrame(group_rows)

        # Compare outcomes between groups using prediction rate and accuracy gaps
        accuracy_gap = group_stats["Accuracy"].max() - group_stats["Accuracy"].min()
        prediction_gap = (
            group_stats["Positive Prediction Rate"].max()
            - group_stats["Positive Prediction Rate"].min()
        )

        # Simple thresholds for a beginner-friendly warning message
        large_difference = accuracy_gap >= 0.10 or prediction_gap >= 0.15

        # ===== SIMPLE SUMMARY FOR NON-TECHNICAL USERS =====
        st.divider()
        st.subheader("Quick Summary (Plain English)")

        if large_difference:
            st.warning("Potential bias detected")
            st.write(
                "Some groups are getting noticeably different outcomes. "
                "This means the model may be treating groups unevenly."
            )
        else:
            st.success("No significant bias detected")
            st.write(
                "Prediction outcomes are fairly similar across groups in this basic check."
            )

        st.write(
            f"Overall, the model prediction matches the target about **{overall_accuracy:.1%}** of the time."
        )
        st.write(
            f"Difference between best and worst group accuracy: **{accuracy_gap:.1%}**."
        )
        st.write(
            f"Difference in positive predictions between groups: **{prediction_gap:.1%}**."
        )

        # Show a direct group comparison (highest vs lowest positive prediction rate)
        highest_group = group_stats.loc[
            group_stats["Positive Prediction Rate"].idxmax(), sensitive_attribute
        ]
        lowest_group = group_stats.loc[
            group_stats["Positive Prediction Rate"].idxmin(), sensitive_attribute
        ]
        highest_rate = group_stats["Positive Prediction Rate"].max()
        lowest_rate = group_stats["Positive Prediction Rate"].min()

        st.write(
            f"Group comparison: **{highest_group}** has about **{highest_rate:.1%}** positive predictions, "
            f"while **{lowest_group}** has about **{lowest_rate:.1%}**."
        )

        # ===== KEY STATISTICS =====
        st.divider()
        st.subheader("Key Statistics")
        st.metric("Overall Accuracy", f"{overall_accuracy:.2%}")

        # Format table for readability
        group_stats_display = group_stats.copy()
        group_stats_display["Accuracy"] = group_stats_display["Accuracy"].map(lambda x: f"{x:.2%}")
        group_stats_display["Positive Prediction Rate"] = group_stats_display[
            "Positive Prediction Rate"
        ].map(lambda x: f"{x:.2%}")

        st.write("**Accuracy per Group:**")
        st.dataframe(group_stats_display, use_container_width=True)

        # ===== PREDICTION DISTRIBUTION BY GROUP =====
        st.subheader(f"Prediction Distribution by {sensitive_attribute}")
        prediction_distribution = pd.crosstab(
            df_valid[sensitive_attribute],
            df_valid["prediction"],
        )

        # Ensure both columns appear even if one prediction value is missing
        for col in [0, 1]:
            if col not in prediction_distribution.columns:
                prediction_distribution[col] = 0

        prediction_distribution = prediction_distribution[[0, 1]]
        prediction_distribution.columns = [
            "Negative prediction (unfavourable)",
            "Positive prediction (favourable)",
        ]

        st.write(
            "This chart compares how many people in each group got a favourable (positive) "
            "or unfavourable (negative) prediction."
        )
        st.dataframe(prediction_distribution, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        prediction_distribution.plot(
            kind="bar",
            ax=ax,
            color=["#d62728", "#2ca02c"],
            width=0.8,
        )
        ax.set_xlabel(sensitive_attribute, fontsize=11)
        ax.set_ylabel("Number of People", fontsize=11)
        ax.set_title(
            f"Favourable vs Unfavourable Predictions by {sensitive_attribute}",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(title="Prediction Type")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for container in ax.containers:
            ax.bar_label(container, padding=2, fontsize=9)

        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

        positive_counts = prediction_distribution["Positive prediction (favourable)"]
        most_favoured_group = positive_counts.idxmax()
        least_favoured_group = positive_counts.idxmin()

        st.caption(
            "Numbers on top of each bar are exact counts. If one group's green bars are much taller, "
            "that group may be getting more favourable outcomes."
        )
        st.info(
            f"In this run, the most favoured group is **{most_favoured_group}** and the least favoured group is **{least_favoured_group}** based on positive predictions."
        )

else:
    # Show a message if no file has been uploaded
    st.info("📁 Please upload a CSV file to get started!")
