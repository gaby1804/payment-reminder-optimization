import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import scipy.stats as stats
from pathlib import Path

# Configuration dictionary for data generation parameters
config = {
    "n_customers": 5000,  # Approximate number of clients
    "start_year": 2022,
    "end_year": 2025,
    "delinquency_rates": {"low": 0.05, "medium": 0.15, "high": 0.30},
    "output_dir": "data/raw",
    "random_state": 41
}

# Set random seeds for reproducibility
np.random.seed(config["random_state"])
fake = Faker()
Faker.seed(config["random_state"])

# List of all possible states for faster selection
ALL_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]


def generate_risk_tier(n):
    """
    Generates risk tiers for n customers.
    Already vectorized using NumPy.
    """
    return np.random.choice(["low", "medium", "high"], size=n, p=[0.6, 0.3, 0.1])


def generate_credit_scores(risk_tiers):
    """
    Generates credit scores based on risk tiers.
    Already vectorized using NumPy.
    """
    scores = np.zeros(len(risk_tiers), dtype=int)
    low_mask = risk_tiers == "low"
    medium_mask = risk_tiers == "medium"
    high_mask = risk_tiers == "high"

    # Generate scores for each tier using normal distribution
    scores[low_mask] = stats.norm.rvs(loc=750, scale=50, size=np.sum(low_mask))
    scores[medium_mask] = stats.norm.rvs(loc=650, scale=75, size=np.sum(medium_mask))
    scores[high_mask] = stats.norm.rvs(loc=550, scale=100, size=np.sum(high_mask))

    # Clip scores to a valid range (300-850)
    return np.clip(scores, 300, 850).astype(int)


def generate_customers(n):
    """
    Generates customer data in a more vectorized way.
    Optimized: Replaced loop for signup_dates with vectorized date generation.
    Used np.random.choice for state to avoid Faker loop.
    """
    risk_tiers = generate_risk_tier(n)

    # Vectorized signup date generation
    start_date_obj = datetime(config["start_year"] - 3, 1, 1)
    end_date_obj = datetime(config["start_year"], 12, 31)
    # Calculate total days in the period
    total_days = (end_date_obj - start_date_obj).days
    # Generate random days to add to the start date
    random_days = np.random.randint(0, total_days + 1, size=n)
    signup_dates = [
        (start_date_obj + timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in random_days
    ]

    data = {
        "customer_id": [f"CUST_{i:05d}" for i in range(n)],
        "signup_date": signup_dates,
        "risk_tier": risk_tiers,
        "credit_score": generate_credit_scores(risk_tiers),
        # Generating emails and phones still requires looping Faker, as it's not easily vectorized for unique values
        "email": [fake.email() for _ in range(n)],
        "phone": [fake.phone_number() for _ in range(n)],
        # Use np.random.choice for states for better performance
        "state": np.random.choice(ALL_STATES, size=n),
        "income_bracket": np.random.choice(["low", "middle", "high"], size=n, p=[0.3, 0.5, 0.2])
    }
    return pd.DataFrame(data)


def generate_accounts(customers_df):
    """
    Generates account data, highly optimized to avoid iterrows.
    Optimized: Used vectorized operations and DataFrame expansion.
    FIXED: TypeError in delta_days calculation.
    """
    account_types = ["credit", "loan", "mortgage"]
    account_type_probs = [0.7, 0.2, 0.1]
    base_limits = {"credit": 5000, "loan": 20000, "mortgage": 300000}
    income_multipliers = {"low": 0.8, "middle": 1.0, "high": 1.5}

    # Determine number of accounts per customer
    n_accounts_per_customer = np.random.randint(1, 4, size=len(customers_df))
    customers_df = customers_df.assign(n_accounts=n_accounts_per_customer)

    # Expand the DataFrame based on the number of accounts
    # This creates a row for each account for each customer
    accounts_expanded_df = customers_df.loc[
        customers_df.index.repeat(customers_df["n_accounts"])
    ].reset_index(drop=True)

    # Generate account IDs
    # Generate a counter for each customer's accounts
    accounts_expanded_df['account_counter'] = accounts_expanded_df.groupby('customer_id').cumcount()
    accounts_expanded_df['account_id'] = accounts_expanded_df.apply(
        lambda row: f"ACC_{row['customer_id']}_{row['account_counter']}", axis=1
    )

    # Vectorized generation of account types
    accounts_expanded_df["account_type"] = np.random.choice(
        account_types, size=len(accounts_expanded_df), p=account_type_probs
    )

    # Vectorized credit limit calculation
    # Map income bracket to multiplier for all rows
    multipliers = accounts_expanded_df["income_bracket"].map(income_multipliers).values
    # Map account type to base limit for all rows
    base_limits_arr = accounts_expanded_df["account_type"].map(base_limits).values
    # Generate Pareto distributed factors
    pareto_factors = stats.pareto.rvs(1.16, size=len(accounts_expanded_df))
    # Calculate credit limit
    accounts_expanded_df["credit_limit"] = (
                base_limits_arr * multipliers * (1 + pareto_factors)
    ).round(2)

    # Vectorized open date generation
    # Convert signup_date to datetime objects once
    signup_dates_dt = pd.to_datetime(accounts_expanded_df["signup_date"])
    # Convert end_date_obj to pd.Timestamp for consistent arithmetic
    end_date_obj_ts = pd.Timestamp(datetime(config["start_year"] + 1, 12, 31).date())

    # Calculate days difference for each customer's signup date up to end_date
    # Fixed: Use pd.Timestamp for end_date_obj for correct subtraction
    delta_days = (end_date_obj_ts - signup_dates_dt).dt.days.values

    # Generate random days within the valid range for each row
    # This loop is necessary as `np.random.randint` needs a scalar high value or matching array,
    # and the upper bound `d + 1` varies for each row.
    random_days_add = np.array([np.random.randint(0, d + 1) for d in delta_days])

    accounts_expanded_df["open_date"] = (
                signup_dates_dt + pd.to_timedelta(random_days_add, unit='D')
    ).dt.strftime("%Y-%m-%d")

    accounts_expanded_df["status"] = "active"

    return accounts_expanded_df[[
        "account_id", "customer_id", "account_type", "credit_limit", "open_date", "status"
    ]]


def generate_payment_schedules(accounts_df):
    """
    Generates payment schedules, highly optimized to avoid iterrows.
    Optimized: Uses cross-merge to create all potential schedules and then filters.
    """
    if accounts_df.empty:
        return pd.DataFrame(columns=["schedule_id", "account_id", "due_date", "amount_due", "is_paid"])

    # Prepare accounts data with datetime objects
    accounts_df['open_date_dt'] = pd.to_datetime(accounts_df['open_date'])
    accounts_df['open_year'] = accounts_df['open_date_dt'].dt.year
    accounts_df['open_month'] = accounts_df['open_date_dt'].dt.month

    # Generate all months from start_year to end_year
    all_months_dt = pd.date_range(
        start=f'{config["start_year"]}-01-01',
        end=f'{config["end_year"]}-12-01',
        freq='MS'  # Month start frequency
    )
    all_months_df = pd.DataFrame({'schedule_date_month_start': all_months_dt})
    all_months_df['schedule_year'] = all_months_df['schedule_date_month_start'].dt.year
    all_months_df['schedule_month'] = all_months_df['schedule_date_month_start'].dt.month

    # Perform a cross merge to get all combinations of accounts and months
    # Pandas 1.2+ supports how='cross' directly. If an older version is used,
    # add a dummy key to both DFs and really merge on that key.
    # Here, we'll assume a recent enough Pandas version or use dummy key if needed.
    merged_schedules = pd.merge(accounts_df, all_months_df, how='cross')

    # Filter out schedules that are before the account's open date or after the end year
    merged_schedules = merged_schedules[
        ((merged_schedules['schedule_year'] > merged_schedules['open_year']) |
         ((merged_schedules['schedule_year'] == merged_schedules['open_year']) &
          (merged_schedules['schedule_month'] >= merged_schedules['open_month']))) &
        (merged_schedules['schedule_year'] <= config['end_year'])
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Calculate due_date (15th of the month)
    # Create due_date as a datetime object directly for calculations
    merged_schedules['due_date_dt'] = pd.to_datetime(
        merged_schedules['schedule_date_month_start'].dt.strftime('%Y-%m-15')
    )
    merged_schedules['due_date'] = merged_schedules['due_date_dt'].dt.strftime("%Y-%m-%d")

    # Vectorized calculation of amount_due
    months = merged_schedules["schedule_month"].values
    years = merged_schedules["schedule_year"].values

    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (months - 11) / 12)
    economic_factor = 1 + 0.03 * (years - 2022)
    merged_schedules["amount_due"] = (
                merged_schedules["credit_limit"] * 0.2 * seasonal_factor * economic_factor
    ).round(2)

    # Generate schedule IDs
    merged_schedules["schedule_id"] = merged_schedules.apply(
        lambda row: f"{row['account_id']}_{row['schedule_year']}_{row['schedule_month']:02d}", axis=1
    )
    merged_schedules["is_paid"] = None  # Initialize as None

    return merged_schedules[[
        "schedule_id", "account_id", "due_date", "amount_due", "is_paid"
    ]]


def generate_payments(schedules_df_input, customers_df, accounts_df):
    """
    Generates payments and identifies reminder candidates.
    FIXED: Now modifies the passed schedules_df_input directly and returns it.
    """
    # Create a local copy to work with, to ensure we don't modify the original
    # if it's used elsewhere for other purposes *before* this function returns.
    # However, since we explicitly return it now, this copy is the one that will be used.
    schedules_df = schedules_df_input.copy()

    # Merge required dataframes once
    merged = schedules_df.merge(
        accounts_df[["account_id", "customer_id"]],
        on="account_id",
        how="left"  # Use left merge to keep all schedules
    ).merge(
        customers_df[["customer_id", "risk_tier"]],
        on="customer_id",
        how="left"  # Use left merge to keep all schedules
    )

    # Convert to datetime for calculations
    due_dates_dt = pd.to_datetime(merged["due_date"])
    years = due_dates_dt.dt.year

    # Calculate delinquency probabilities in a vectorized manner
    risk_factors = merged["risk_tier"].map(config["delinquency_rates"]).values
    economic_factors = 1 + 0.1 * (years - 2022)
    delinquency_probs = np.minimum(0.95, risk_factors * economic_factors)
    # Determine which schedules are paid based on probability
    is_paid_mask = np.random.rand(len(merged)) > delinquency_probs

    # Correct way to update 'is_paid' in schedules_df:
    # Use .loc for setting values to avoid SettingWithCopyWarning and ensure direct modification
    schedules_df.loc[:, 'is_paid'] = is_paid_mask

    # Select only paid schedules for payment generation
    paid_schedules_merged = merged[is_paid_mask].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Calculate days late for paid schedules
    scales = 7 * (1 + 0.15 * (
                years[is_paid_mask].values - 2022))  # This indexing is fine as `years` and `is_paid_mask` align.
    days_late = stats.weibull_min.rvs(0.8, loc=0, scale=scales, size=len(paid_schedules_merged))
    days_late_int = np.maximum(0, days_late).astype(int)

    # Calculate payment dates
    payment_dates_dt = due_dates_dt[is_paid_mask] + pd.to_timedelta(days_late_int, unit='D')

    # Create payments DataFrame directly from arrays
    payments_data = {
        "payment_id": ["PMT_" + sid for sid in paid_schedules_merged["schedule_id"]],
        "schedule_id": paid_schedules_merged["schedule_id"].values,
        "payment_date": payment_dates_dt.dt.strftime("%Y-%m-%d").values,
        "amount_paid": (paid_schedules_merged["amount_due"] *
                        np.random.beta(2, 0.5, size=len(paid_schedules_merged))).round(2).values,
        "days_late": days_late_int,
        "payment_method": np.random.choice(["auto-pay", "manual", "bank_transfer"],
                                           size=len(paid_schedules_merged), p=[0.4, 0.5, 0.1]),
        "customer_id": paid_schedules_merged["customer_id"].values
    }

    payments_df = pd.DataFrame(payments_data)

    # Identify reminder candidates (unpaid schedules)
    reminder_candidates = merged.loc[~is_paid_mask, "schedule_id"].tolist()

    return payments_df, reminder_candidates, schedules_df # Return the updated schedules_df


def generate_reminders(schedules_df, reminder_candidates):
    """
    Generates reminder data, optimized to avoid iterrows.
    Optimized: Creates an expanded DataFrame for reminders and applies vectorized operations.
    """
    if not reminder_candidates:
        return pd.DataFrame(columns=[
            "reminder_id", "account_id", "sent_at", "channel",
            "opened", "clicked", "payment_triggered", "year"
        ])

    # Filter schedules for reminder candidates
    candidate_schedules = schedules_df[schedules_df["schedule_id"].isin(reminder_candidates)].copy()
    candidate_schedules["due_date_dt"] = pd.to_datetime(candidate_schedules["due_date"])

    # Determine number of reminders per candidate
    n_reminders_per_candidate = np.random.randint(1, 5, size=len(candidate_schedules))
    candidate_schedules = candidate_schedules.assign(n_reminders=n_reminders_per_candidate)

    # Expand the DataFrame based on the number of reminders
    reminders_expanded_df = candidate_schedules.loc[
        candidate_schedules.index.repeat(candidate_schedules["n_reminders"])
    ].reset_index(drop=True)

    # Generate reminder index for each schedule_id
    reminders_expanded_df['reminder_idx'] = reminders_expanded_df.groupby('schedule_id').cumcount()

    # Generate reminder IDs
    reminders_expanded_df["reminder_id"] = reminders_expanded_df.apply(
        lambda row: f"REM_{row['schedule_id']}_{row['reminder_idx']}", axis=1
    )

    # Calculate sent_at times
    # Days to add for each reminder based on its index
    days_to_add = (reminders_expanded_df["reminder_idx"] * 2 - 2).values
    reminders_expanded_df["sent_at_dt"] = (
                reminders_expanded_df["due_date_dt"] + pd.to_timedelta(days_to_add, unit='D')
    )
    reminders_expanded_df["sent_at"] = reminders_expanded_df["sent_at_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Extract year for calculations
    years = reminders_expanded_df["due_date_dt"].dt.year.values

    # Vectorized calculations for opened, clicked, payment_triggered
    tech_improvement = 1 + 0.05 * (years - 2022)
    base_rate_factor = 0.65 ** reminders_expanded_df["reminder_idx"].values
    base_rate = 0.7 * tech_improvement * base_rate_factor

    opened = np.random.rand(len(reminders_expanded_df)) < base_rate
    clicked = opened & (np.random.rand(len(reminders_expanded_df)) < (0.4 * tech_improvement))
    triggered = clicked & (np.random.rand(len(reminders_expanded_df)) < (0.6 * tech_improvement))

    reminders_expanded_df["opened"] = opened
    reminders_expanded_df["clicked"] = clicked
    reminders_expanded_df["payment_triggered"] = triggered

    # Vectorized channel selection
    reminders_expanded_df["channel"] = np.random.choice(
        ["email", "sms", "push"],
        size=len(reminders_expanded_df),
        p=[0.5, 0.4, 0.1]
    )
    reminders_expanded_df["year"] = years

    return reminders_expanded_df[[
        "reminder_id", "account_id", "sent_at", "channel",
        "opened", "clicked", "payment_triggered", "year"
    ]]


def generate_feedback(customers_df, payments_df):
    """
    Generates feedback data, highly optimized to avoid iterrows.
    Optimized: Filters customers first and then applies vectorized calculations.
    """
    # Calculate total late days per customer
    late_payments = payments_df.groupby("customer_id")["days_late"].sum().reset_index()

    # Determine which customers will provide feedback (25% chance)
    feedback_mask = np.random.rand(len(customers_df)) < 0.25
    feedback_customers_df = customers_df[feedback_mask].copy()

    if feedback_customers_df.empty:
        return pd.DataFrame(columns=[
            "feedback_id", "customer_id", "survey_date",
            "satisfaction_score", "complaint_topic"
        ])

    # Merge with late payments data
    feedback_customers_df = feedback_customers_df.merge(
        late_payments, on="customer_id", how="left"
    ).fillna({"days_late": 0})  # Fill NaN for customers with no late payments

    # Vectorized satisfaction score calculation
    base_score_map = {"low": 4, "medium": 3, "high": 2}
    base_scores = feedback_customers_df["risk_tier"].map(base_score_map).values
    late_days_effect = (feedback_customers_df["days_late"] / 60).astype(int).values
    random_adjustment = np.random.randint(-1, 2, size=len(feedback_customers_df))

    satisfaction = np.maximum(1, np.minimum(5, base_scores - late_days_effect + random_adjustment))
    feedback_customers_df["satisfaction_score"] = satisfaction

    # Vectorized complaint topic selection
    feedback_customers_df["complaint_topic"] = np.random.choice(
        ["none", "timing", "frequency", "clarity"],
        size=len(feedback_customers_df),
        p=[0.5, 0.3, 0.15, 0.05]
    )

    # Vectorized survey date generation
    start_date_obj = datetime(config['start_year'], 1, 1)
    end_date_obj = datetime(config['end_year'], 12, 31)
    total_days_survey = (end_date_obj - start_date_obj).days
    random_days_survey = np.random.randint(0, total_days_survey + 1, size=len(feedback_customers_df))
    feedback_customers_df["survey_date"] = (
                pd.Series([start_date_obj] * len(feedback_customers_df)) + pd.to_timedelta(random_days_survey, unit='D')
    ).dt.strftime("%Y-%m-%d")

    # Generate feedback IDs
    # Generate unique IDs using UUID for robustness, as customer_id isn't enough for feedback_id
    feedback_customers_df["feedback_id"] = feedback_customers_df["customer_id"].apply(
        lambda cid: f"FB_{cid}_{fake.uuid4()[:8]}"
    )

    return feedback_customers_df[[
        "feedback_id", "customer_id", "survey_date", "satisfaction_score", "complaint_topic"
    ]]


print("Generating customer data...")
customers_df = generate_customers(config["n_customers"])
print("Generating account data...")
accounts_df = generate_accounts(customers_df)
print("Generating payment schedules...")
# We pass a copy of accounts_df to prevent modifying the original DataFrame in generate_payment_schedules
# if it were to add temporary columns, though with the new vectorized approach, it's less critical.
schedules_df = generate_payment_schedules(accounts_df.copy())
print("Generating payments and reminder candidates...")
# CAPTURE THE UPDATED schedules_df HERE
payments_df, reminder_candidates, schedules_df = generate_payments(schedules_df, customers_df, accounts_df)
print("Generating reminders...")
reminders_df = generate_reminders(schedules_df, reminder_candidates)
print("Generating feedback data...")
feedback_df = generate_feedback(customers_df, payments_df)

# SAVE FILES
print("Saving data to CSV files...")
output_dir = Path(config["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

customers_df.to_csv(output_dir / "customers.csv", index=False)
accounts_df.to_csv(output_dir / "accounts.csv", index=False)
schedules_df.to_csv(output_dir / "payment_schedules.csv", index=False)
payments_df.to_csv(output_dir / "payments.csv", index=False)
reminders_df.to_csv(output_dir / "reminders.csv", index=False)
feedback_df.to_csv(output_dir / "feedback.csv", index=False)
print("Data generation complete!")

