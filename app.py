import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight

# Title 
st.title("Welcome To FairChecker, Your Personal AI Bias Detection Tool")

# Short description
st.write(
    "Analyse your machine learning dataset for fairness across demographic groups "
    "NO coding knowledge required - just upload your data and explore the results."
)

# Quick way to load example datasets without needing to upload anything
st.subheader("Quick Load a Sample Dataset")
st.write("Click a button below to instantly load a well-known dataset.")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Load Adult Income Data"):
        df_preset = pd.read_csv("datasets/adult_train_clean.csv")
        st.session_state["preloaded_df"] = df_preset
        st.session_state["dataset_name"] = "Adult Income"
        st.success("Adult Income dataset ready. This dataset contains US census records. We will check whether income predictions are fair across gender groups.")

with col2:
    if st.button("Load COMPAS Recidivism Data"):
        df_preset = pd.read_csv("datasets/compas_clean.csv")
        st.session_state["preloaded_df"] = df_preset
        st.session_state["dataset_name"] = "COMPAS"
        st.success("COMPAS Recidivism dataset ready. This dataset contains US criminal justice data. We will check whether reoffending predictions are fair across racial groups.")

with col3:
    if st.button("Load Heart Disease Data"):
        cols = ["age","sex","cp","trestbps","chol","fbs",
                "restecg","thalach","exang","oldpeak","slope","ca",
                "thal","target"]
        df_preset = pd.read_csv("datasets/heart.data", header=None, names=cols, sep=",")
        df_preset = df_preset.replace("?", float("nan"))
        st.session_state["preloaded_df"] = df_preset
        st.session_state["dataset_name"] = "Heart Disease"
        st.success("Heart Disease dataset ready. This dataset contains Cleveland heart disease records. We will check whether predictions are fair across gender groups.")

st.divider()

# upload own dataset
uploaded_file = st.file_uploader("Or upload your own CSV file:", type="csv")

# Load dataset from uploaded file or prev stored session state
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif "preloaded_df" in st.session_state:
    df = st.session_state["preloaded_df"]

if df is not None:
    
    # first 5 rows of the dataset
    st.subheader("Preview of Your Dataset (First 5 Rows)")
    st.dataframe(df.head())
    
    # Button to show example profile from the uploaded dataset
    if st.button("Generate Random Data"):
        columns_lower = [col.lower() for col in df.columns]

        #decide if this is the adult income dataset by looking for specific column names and show example profiles 
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

            #pick a random example profile to show what the data looks like
            selected_profile = random.choice(profile_options)

            st.info("Here is an example profile from this dataset:")
            st.table(pd.DataFrame(selected_profile, columns=["Field", "Value"]))
            
            st.caption(
                "This is an example profile from the dataset, showing what a typical record looks like in plain English."
            )


        #detect COMPAS dataset by looking for specific column names and show example profiles
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

            #Pick a random example profile to show what the data looks like
            selected_profile = random.choice(profile_options)

            st.info("Here is an example profile from this dataset:")
            st.table(pd.DataFrame(selected_profile, columns=["Field", "Value"]))
            
            st.caption(
                "This is an example profile from the dataset, showing what a typical record looks like in plain English."
            )

        #if a heart disease dataset is detected, show example profiles
        elif any("thalach" in col.lower() for col in columns_lower):
            profile_options = [
                [
                    ["Age", "58"],
                    ["Sex", "Male"],
                    ["Chest Pain Type", "Typical"],
                    ["Max Heart Rate", "140"],
                    ["Heart Disease", "Present"],
                ],
                [
                    ["Age", "45"],
                    ["Sex", "Female"],
                    ["Chest Pain Type", "Non-anginal"],
                    ["Max Heart Rate", "168"],
                    ["Heart Disease", "Absent"],
                ],
                [
                    ["Age", "63"],
                    ["Sex", "Male"],
                    ["Chest Pain Type", "Asymptomatic"],
                    ["Max Heart Rate", "132"],
                    ["Heart Disease", "Present"],
                ],
            ]

            #Pick a random example profile tp show what the data looks like 
            selected_profile = random.choice(profile_options)

            st.info("Here is an example profile from this dataset:")
            st.table(pd.DataFrame(selected_profile, columns=["Field", "Value"]))

            st.caption(
                "This shows a sample record from the dataset, showing what a typical record looks like in a simple format."
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
                "credit",
                "target",
            ]
            target_for_card = None
            for name in likely_target_names:
                matches = [col for col in df.columns if col.lower() == name.lower()]
                if matches:
                    target_for_card = matches[0]
                    break

            # Pick readable columns for profile
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

                # skip binary columns unless this is the likely target
                if is_binary_only and col != target_for_card:
                    continue

                readable_cols.append(col)
                if len(readable_cols) >= 4:
                    break

            # choose 3-4 columns o show in the profile card 
            readable_cols = readable_cols[:4]

            #if there are too few reqadable columns, add a few extra
            if len(readable_cols) < 3:
                for col in df.columns:
                    if col not in readable_cols:
                        readable_cols.append(col)
                    if len(readable_cols) >= 3:
                        break

            #build small profile tabe
            profile_table = pd.DataFrame(
                {
                    "Field": readable_cols,
                    "Value": [sample_row[col] for col in readable_cols],
                }
            )

            st.info("Here is an example profile from this dataset:")
            st.table(profile_table)
            st.caption(
                "This is an example profile from the dataset, showing what a typical record looks like in a simple format."
            )
    
    # Get the list of column names from loaded dataset
    columns = df.columns.tolist()


    # Check which dataset it is to pre-fill dropdown
    #convert to lowercase so matching is not case-sensitive
    columns_lower = [col.lower() for col in columns]
    detected_dataset = None
    default_target_column = columns[0] if columns else None
    default_sensitive_attribute = columns[0] if columns else None



    # looks to find a column by exact name (case-insensitive)
    def find_column_name(possible_names):
        for name in possible_names:
            for col in columns:
                if col.lower() == name.lower():
                    return col
        return None

    # Adult set has income or salary column
    has_income = any("income" in col for col in columns_lower)
    has_salary = any("salary" in col for col in columns_lower)

    # Compas has specific recidivism column name
    compas_target = find_column_name(["Two_yr_Recidivism", "two_year_recid"])

    # Heart Disease has both target and thalach (max heart rate) columns
    has_target = any(col == "target" for col in columns_lower)
    has_thalach = any(col == "thalach" for col in columns_lower)

    if compas_target is not None:
        detected_dataset = "COMPAS"

        # If race is split across one-hot columns, rebuild into a single 'race' column 
        race_column = find_column_name(["race"])
        possible_race_columns = [
            "African_American",
            "Asian",
            "Caucasian",
            "Hispanic",
            "Native_American",
            "Other",
        ]

        #Check which of these race columns actually exist in the dataset
        existing_race_one_hot = [col for col in possible_race_columns if col in df.columns]
        
        #Extra check in case column names are slightly different ie, lowercase or missing underscores (eg, "african american" instead of "African_American")
        has_african_american_col = any("african_american" in col.lower() for col in df.columns)

        #If no single race column exists but one-hot columns do, rebuild it
        if race_column is None and has_african_american_col and existing_race_one_hot:
            one_hot_data = df[existing_race_one_hot]

            #Pick the race with the highest value in the one-hot columns for each row to create
            df["race"] = one_hot_data.idxmax(axis=1)

            #Handle rows where no race was recorded
            df.loc[one_hot_data.sum(axis=1) == 0, "race"] = "Unknown"

            #Update column list after adding race
            columns = df.columns.tolist()

        #Set sensible defaults for target and sensitive attribute columns based COMPAS column names 
        default_target_column = "Two_yr_Recidivism" if "Two_yr_Recidivism" in columns else compas_target
        default_sensitive_attribute = "race" if "race" in df.columns else columns[0]
        st.success("COMPAS dataset detected. Columns have been automatically configured for you")

    elif has_target and has_thalach:
        detected_dataset = "Heart Disease"
        default_target_column = "target"
        default_sensitive_attribute = "sex"
        st.success("Heart Disease dataset detected. Columns have been automatically configured for you")

    elif has_income or has_salary:
        detected_dataset = "Adult Income"
        default_target_column = find_column_name(["income", "salary"]) or columns[0]
        default_sensitive_attribute = find_column_name(["sex", "gender", "race"]) or columns[0]
        st.success("Adult Income dataset detected. Columns have been automatically configured for you")

    else:
        st.info("Dataset not recognised. Please select your target and sensitive attribute columns manually")

    # Set default dropdown if column exists
    target_default_index = columns.index(default_target_column) if default_target_column in columns else 0
    sensitive_default_index = (
        columns.index(default_sensitive_attribute) if default_sensitive_attribute in columns else 0
    )
    
    # Create two columns for the dropdowns side by side
    col1, col2 = st.columns(2)
    
    # Dropdown 1: Select target column
    with col1:
        target_column = st.selectbox(
            "Select Target Column",
            columns,
            index=target_default_index,
            help="Choose the column you want to predict or analyse"
        )
    
    # Dropdown 2: Select sensitive attribute
    with col2:
        sensitive_attribute = st.selectbox(
            "Select Sensitive Attribute Column",
            columns,
            index=sensitive_default_index,
            help="Choose the demographic or sensitive column (e.g., age, gender, race)"
        )

    # Automatically make a numeric version of the selected target column
    # DO NOT change the original target column 
    # ie, "<=50K" -> 0 and ">50K" -> 1

    df["target_numeric"] = (
        df[target_column]
        .astype(str)
        .str.strip()
        .replace({"<=50K": 0, ">50K": 1})
    )
    df["target_numeric"] = pd.to_numeric(
        df["target_numeric"], errors="coerce")
    
    # Button to run basic checks and bias analysis
    if st.button("Run Basic Check", use_container_width=True):
        st.subheader("Analysis Results")

        # Show the selected columns and dataset size
        st.write(f"**Target Column:** `{target_column}`")
        st.write(f"**Sensitive Attribute:** `{sensitive_attribute}`")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Rows", df.shape[0])
        with col2:
            st.metric("Number of Columns", df.shape[1])

        # BIAS ANALYSIS SECTION 
        # Create a copy so the og uploaded data isnt changed by he analysis
        df_analysis = df.copy()

        # Use target_numeric for calculation
        valid_rows = df_analysis["target_numeric"].isin([0, 1])

        if valid_rows.sum() == 0:
            st.error("Could not create valid binary target values (0/1) from the selected target column.")
            st.info("Try a target column with values like 0/1 or <=50K/>50K.")
            st.stop()

        # Keep valid rows for model training and fairness calc
        df_valid = df_analysis[valid_rows].copy()
        df_valid["target_numeric"] = df_valid["target_numeric"].astype(int)

        # Decision Tree classifier training for predictions
        df_encoded = df_valid.copy()
        le_dict = {}

        # Encode any categorical columns so the model can work with them 
        for col in df_encoded.columns:
            if df_encoded[col].dtype == object and col not in [target_column, sensitive_attribute]:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                le_dict[col] = le

        # Get features ready for decision tree (no inc target+sensitive stuff)
        feature_cols = [
    c for c in df_encoded.columns
    if c not in [
        target_column,
        "target_numeric",
        sensitive_attribute,
        "prediction",
    ]
]

        if len(feature_cols) > 0 and len(df_encoded) > 1:
            X = df_encoded[feature_cols].copy()
            y = df_encoded["target_numeric"]

            # Keep track of og row indexes to link predictions back later for fairness calc
            row_indices = df_encoded.index
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X,
                y,
                row_indices,
                test_size=0.2,
                random_state=24,
            )

            # 80/20 split train on training data, then predict on unseen test data
            max_depth = 5
            dt = DecisionTreeClassifier(
                max_depth=max_depth,
                random_state=24,
                min_samples_split=5)
            dt.fit(X_train, y_train)
            test_predictions = dt.predict(X_test)
            y_pred = test_predictions

            st.success("Model successfully trained and ready")

            # Only use the test set rows for fairness calcs (don't evaluate on training data)
            df_valid_all = df_valid.copy()
            test_indices = idx_test
            df_valid = df_valid.loc[test_indices].copy()
            df_valid["prediction"] = test_predictions
        else:
            # If DT doesn't have enough features to train model, fall back to simple predictions 
            df_valid["prediction"] = (df_valid["target_numeric"].mean() > 0.5).astype(int)

        # Calculate overall accuracy of model
        overall_accuracy = (df_valid["prediction"] == df_valid["target_numeric"]).mean()

        # Now break results down by each group in the sensitive attribute to see if some groups are treated differently by the model than others
        # (THIS IS WHERE WE WILL CHECK FOR BIAS)
        group_rows = []
        for group_value, group_data in df_valid.groupby(sensitive_attribute):
            
            #Accuracy in this specfic group
            group_accuracy = (group_data["prediction"] == group_data["target_numeric"]).mean()

            #How often the model predicts the psostive outcome (ie, >50k)
            positive_rate = (group_data["prediction"] == 1).mean()

            group_rows.append(
                {
                    sensitive_attribute: group_value,
                    "Accuracy": group_accuracy,
                    "Positive Prediction Rate": positive_rate,
                    "Samples": len(group_data),
                }
            )
        #Turn results into a dataframe for easier comparison and display
        group_stats = pd.DataFrame(group_rows)

        # Compare groups by looking at the gaps between best and worst outcomess
        accuracy_gap = group_stats["Accuracy"].max() - group_stats["Accuracy"].min()
        prediction_gap = (
            group_stats["Positive Prediction Rate"].max()
            - group_stats["Positive Prediction Rate"].min()
        )

        # DEMOGRAPHIC PARITY (DP)
        #Check if different groups are recieving positive predictions at sim rates
        dp_rates = {}
        for group, gdf in df_valid.groupby(sensitive_attribute):
            dp_rates[group] = (gdf["prediction"] == 1).mean()

        #Difference bwteen highest and lowest group rates (bigger gap=more potential bias)
        dp_gap = max(dp_rates.values()) - min(dp_rates.values())


        # EQUAL OPPORTUNITY (EO)
        #Focuses on people who genuinely deserve a positive outcome (ie, target_numeric=1) 
        #Checks if they are identified equally across groups
        eo_rates = {}
        for group, gdf in df_valid.groupby(sensitive_attribute):
            pos_mask = gdf["target_numeric"] == 1

            if pos_mask.sum() > 0:
                #True positive rate for this group
                eo_rates[group] = (gdf.loc[pos_mask, "prediction"] == 1).mean()
            else:
                eo_rates[group] = 0.0  #no positive samples in this group
        eo_gap = max(eo_rates.values()) - min(eo_rates.values())

        # EQUALISED ODDS (EQO) (negative side)
        #looks at how the model behaves on the negative class 
        #ie, people who should NOT get a positive outcome (target_numeric=0)

        eqo_rates = {}
        for group, gdf in df_valid.groupby(sensitive_attribute):
            neg_mask = gdf["target_numeric"] == 0
            if neg_mask.sum() > 0:
                eqo_rates[group] = (gdf.loc[neg_mask, "prediction"] == 1).mean()
            else:
                eqo_rates[group] = 0.0
        eqo_gap = max(eqo_rates.values()) - min(eqo_rates.values())

        # Updated to use fairness metric gaps instead of just raw rates to flag big differences
        # match summary with metric summary
        large_difference = (
            dp_gap >= 0.10 or
            eo_gap >= 0.10 or
            eqo_gap >= 0.10
        )

        # SIMPLE SUMMARY FOR NON-TECHNICAL USERS 

        st.divider()
        st.subheader("Quick Summary")

        if large_difference:
            st.warning("Potential bias detected")
            st.write(
                "Some groups are receiving noticeably different outcomes from the model. "
                "This could mean the model is treating certain demographic groups unfairly."
            )
        else:
            st.success("No significant bias detected")
            st.write(
                "The model's predictions are fairly similar across groups in this basic check."
            )


        #Show overall model performance in simple terms
        st.write(
            f"Overall, the model prediction matches the target about **{overall_accuracy:.1%}** of the time."
        )

        #Show the gaps between groups in simple terms
        st.write(
            f"Difference in accuracy between the most and least favoured group: **{accuracy_gap:.1%}**."
        )
        st.write(
            f"Difference in positive predictions between groups: **{prediction_gap:.1%}**."
        )

        # Pick out the groups with the highest and lowest positive prediction rates
        #Makes it easier to show a direct group comparison (highest vs lowest positive prediction rate)
        highest_group = group_stats.loc[
            group_stats["Positive Prediction Rate"].idxmax(), sensitive_attribute
        ]
        lowest_group = group_stats.loc[
            group_stats["Positive Prediction Rate"].idxmin(), sensitive_attribute
        ]
        highest_rate = group_stats["Positive Prediction Rate"].max()
        lowest_rate = group_stats["Positive Prediction Rate"].min()

        # Show a simple comparison between the most and least favoured groups in terms of positive prediction rates
        #Makes it easier for users to see the difference
        st.write(
            f"Most vs least favoured group: **{highest_group}** has about **{highest_rate:.1%}** positive predictions, "
            f"while **{lowest_group}** has about **{lowest_rate:.1%}**."
        )

        # FAIRNESS METRICS 
        # Calculate key fairness measures across groups bases on selected sensitive attribute 
        # Show the gaps between groups for each measure

        dp_rates = {}
        for group, gdf in df_valid.groupby(sensitive_attribute):

            #how often each group gets a positive prediction 
            dp_rates[group] = (gdf["prediction"] == 1).mean()
            
        #gap between highest and lowest group rates (bigger gap=more potential bias)   
        dp_gap = max(dp_rates.values()) - min(dp_rates.values())

        eo_rates = {}
        for group, gdf in df_valid.groupby(sensitive_attribute):
            pos_mask = gdf["target_numeric"] == 1

            if pos_mask.sum() > 0:
                #among those who should get a positive outcome, how many actually get it in each group (true positive rate)
                eo_rates[group] = (gdf.loc[pos_mask, "prediction"] == 1).mean()
            else:
                eo_rates[group] = 0.0 # no positive samples in this group
        eo_gap = max(eo_rates.values()) - min(eo_rates.values())

        eqo_rates = {}
        for group, gdf in df_valid.groupby(sensitive_attribute):
            neg_mask = gdf["target_numeric"] == 0

            if neg_mask.sum() > 0:
                #among those who should NOT get a positive outcome, how many are incorrectly predicted as positive in each group (false positive rate)
                eqo_rates[group] = (gdf.loc[neg_mask, "prediction"] == 1).mean()
            else:
                eqo_rates[group] = 0.0
        eqo_gap = max(eqo_rates.values()) - min(eqo_rates.values())

        st.divider()
        st.subheader("Fairness Metrics")
        st.write("These three measures check whether the model treats all groups equally. Lower gaps mean fairer predictions.")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Demographic Parity Gap", f"{dp_gap:.1%}")
            st.caption("Do different groups receive positive predictions at equal rates? A gap below 10% is generally considered fair.")
        with m2:
            st.metric("Equal Opportunity Gap", f"{eo_gap:.1%}")
            st.caption("Among those who should receive a positive outcome, are groups treated equally?")
        with m3:
            st.metric("Equalised Odds (Negative) Gap", f"{eqo_gap:.1%}")
            st.caption("Among those who should not receive a positive outcome, is the error rate similar across groups?")

       
        #Interpret fairness gaps using simple thresholds 
        #smaller gaps= more fair,consistent treatment of groups, bigger gaps = more potential bias
        if dp_gap < 0.10 and eo_gap < 0.10 and eqo_gap < 0.10:
            st.success("The model appears broadly fair across all three fairness measures.")

        elif dp_gap >= 0.20 or eo_gap >= 0.20 or eqo_gap >= 0.20:
            st.error("Warning: Significant bias detected. The model is treating groups very differently on at least one fairness measure.")
        else:
            st.warning("Warning: Moderate bias detected. At least one fairness measure shows a noticeable gap between groups.")


        # PREDICTION DISTRIBUTION BY GROUP 
        st.subheader(f"Prediction Distribution by {sensitive_attribute}")
        prediction_distribution = pd.crosstab(
            df_valid[sensitive_attribute],
            df_valid["prediction"],
        )

        # Make sure both outcomes (0 and 1) are always shown
        for col in [0, 1]:
            if col not in prediction_distribution.columns:
                prediction_distribution[col] = 0

        #Reorder columns for consistency
        prediction_distribution = prediction_distribution[[0, 1]]

        #Rename columns so they're easier to undersand for users
        prediction_distribution.columns = [
            "Negative prediction (unfavourable)",
            "Positive prediction (favourable)",
        ]

        #Explain what the chart is showing before displaying it so users know what to look for
        st.write(
            "This chart shows how predictions are split across groups." 
            "It highlights the number of people who received a positive or negative prediction."
        )

       
        #Create a bar chart to show the number of positive and negative predictions for each group 
        fig, ax = plt.subplots(figsize=(10, 5))
        prediction_distribution.plot(
            kind="bar",
            ax=ax,
            color=["#7F77DD", "#1D9E75"],
            width=0.8,
        )

        #Add a clear title and labels to the chart so users can understand it easily
        ax.set_xlabel(sensitive_attribute, fontsize=11)
        ax.set_ylabel("Number of People", fontsize=11)
        ax.set_title(
            f"Favourable vs Unfavourable Predictions by {sensitive_attribute}",
            fontsize=13,
            fontweight="bold",
        )

        #Clean up the chart styling to make it look nicer and easier to read
        ax.legend(title="Prediction Type")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        #Add values on top of each bar to make it easier for users to see the exact numbers
        for container in ax.containers:
            ax.bar_label(container, padding=2, fontsize=9)

        plt.xticks(rotation=0)
        plt.tight_layout()

        #Display the chart in Streamlit
        st.pyplot(fig)

        #Identify which group is most favoured and least favoured based on the number of positive predictions they receive
        positive_counts = prediction_distribution["Positive prediction (favourable)"]
        most_favoured_group = positive_counts.idxmax()
        least_favoured_group = positive_counts.idxmin()

        #Explain the chart in simple terms and highlight the most and least favoured groups so users can easily understand the implications of the results
        st.caption(
            "Each bar represents the number of predictions per group. "
            "If one group's green bar is much taller, that group may be getting more favourable outcomes."
        )

        #Highlight the groups which stand out 
        st.info(
            f"In this run, the most favoured group is **{most_favoured_group}** and the least favoured group is **{least_favoured_group}** based on positive predictions."
        )

        st.divider()
        st.subheader("Model A vs Model B : Effect of Bias Mitigation")
        st.write(
            "Model A is the original decision tree trained on "
            "unmodified data. Model B applies reweighing : a "
            "technique that adjusts the influence of training samples to give "
            "underrepresented groups more weight during training. "
            "Compare the results below."
        )

        #Only run this comparison if training/test data exists
        if "X_train" in locals() and "X_test" in locals() and "y_train" in locals() and "y_test" in locals() and "y_pred" in locals() and "df_valid_all" in locals():
            
            #Combine sensitive attribute + target label so reweighing can balance both
            combined_labels = [f"{s}_{t}" for s, t in zip(df_valid_all.loc[X_train.index, sensitive_attribute].values, y_train.values)]
            
            #Apply reweighing bias mitigation technique to give more weight to underrepresented groups in the training data
            sample_weights = compute_sample_weight(class_weight='balanced', y=combined_labels)
            
            #Train Model B w/mitigation
            model_b = DecisionTreeClassifier(max_depth=max_depth, random_state=24)
            model_b.fit(X_train, y_train, sample_weight=sample_weights)
            
            #Get predictions from Model B on the same test set to compare with Model A
            y_pred_b = model_b.predict(X_test)

            #Get sensitive attribute values for the test set to compare how each model treats different groups
            test_sens = df_valid_all.loc[X_test.index, sensitive_attribute].values
            groups_unique = list(set(test_sens))


            #Store positive prediction rates for each group in both models to compare how the bias mitigation technique affected the predictions for different groups
            rates_a = {}   #Model A (original data)
            rates_b = {}   #Model B (after reweighing bias mitigation)

            for g in groups_unique:
                mask = test_sens == g

                if mask.sum() > 0:
                    #Calculate the positive prediction rate for this group in both models
                    rates_a[g] = (y_pred[mask] == 1).mean()
                    rates_b[g] = (y_pred_b[mask] == 1).mean()


            #VISUAL COMPARISON OF MODEL A AND MODEL B
            fig_ab, ax_ab = plt.subplots(figsize=(10, 5))
           
            x_pos = range(len(groups_unique))
            width = 0.35   #Width of the bars for each model

            #Bars for model A (original data)
            bars_a = ax_ab.bar(
                [i - width/2 for i in x_pos],
                [rates_a[g] for g in groups_unique],
                width=width,
                label="Model A : original data",
                color="#7F77DD",
                alpha=0.85
            )

            #Bars for model B (after reweighing bias mitigation)
            bars_b = ax_ab.bar(
                [i + width/2 for i in x_pos],
                [rates_b[g] for g in groups_unique],
                width=width,
                label="Model B : after reweighing bias mitigation",
                color="#1D9E75",
                alpha=0.85
            )


            #Add perecentage labels on top of each bar to make it easier for users to see the exact positive prediction rates for each group in both models
            for bar in bars_a:
                ax_ab.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.1%}",
                    ha="center", fontsize=9
                )

            for bar in bars_b:
                ax_ab.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.1%}",
                    ha="center", fontsize=9
                )

            #Label axes clearly so users can understand what the chart is showing and compare the two models easily
            ax_ab.set_xlabel(sensitive_attribute, fontsize=11)
            ax_ab.set_ylabel(
                "% Receiving Positive Prediction", fontsize=11)
            
            #Title to explain what the comparison is showing
            ax_ab.set_title(
                "Positive Prediction Rate : Model A vs Model B",
                fontsize=13, fontweight="bold"
            )

            #Set group labels on x-axis
            ax_ab.set_xticks(list(x_pos))
            ax_ab.set_xticklabels(
                groups_unique, rotation=15, ha="right")
            
            #Add legend and clean up chart styling to make it look nicer and easier to read
            ax_ab.legend()
            ax_ab.spines["top"].set_visible(False)
            ax_ab.spines["right"].set_visible(False)

            plt.tight_layout()

            #Display the comparison chart in Streamlit
            st.pyplot(fig_ab)

            
            #Work out the gaps between groups for each model to compare the fairness of Model A and Model B using the demographic parity gap as the main comparison metric
            #Smaller gaps between groups = more fair, bigger gaps = more potential bias
            gap_a = max(rates_a.values()) - min(rates_a.values())
            gap_b = max(rates_b.values()) - min(rates_b.values())
            
            #Check accuracy for both models to compare fairness and performance
            acc_a = (y_pred == y_test.values).mean()
            acc_b = (y_pred_b == y_test.values).mean()

            #Summarise whether reweighting improved the fairness gap
            st.write(
                f"**Demographic parity gap:** "
                f"Model A = {gap_a:.1%} → "
                f"Model B = {gap_b:.1%} "
                f"{'Improved' if gap_b < gap_a else ('Warning: Gap increased after reweighing' if gap_b > gap_a else 'Warning: No change detected')}"
            )

            #Show wheather accuracy changed after mitigation
            st.write(
                f"**Model accuracy:** "
                f"Model A = {acc_a:.1%} → "
                f"Model B = {acc_b:.1%}"
            )
            
            #Remind users that fairness improvements can affect accuracy
            st.caption(
                "Note: Improving fairness sometimes reduces "
                "accuracy slightly. This is known as the "
                "fairness-accuracy trade-off."
            )

else:
    #Show a message before any file isuploaded
    st.info("Please upload a CSV file to get started!")

