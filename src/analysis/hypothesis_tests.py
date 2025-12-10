import os
import pandas as pd
import numpy as np
from scipy import stats

# -----------------------------
# Load cleaned data
# -----------------------------
df = pd.read_csv("data/MachineLearningRating_clean.csv", low_memory=False)

# Define key metrics
df['claim_occurred'] = np.where(df['totalclaims'] > 0, 1, 0)
df['claim_severity'] = df['totalclaims']

# Function to compute claim frequency
def claim_frequency(group):
    return group['claim_occurred'].mean()

# Function to compute claim severity
def claim_severity(group):
    return group.loc[group['claim_occurred'] == 1, 'claim_severity'].mean()

# -----------------------------
# Hypothesis 1: Risk differences across provinces
# -----------------------------
provinces = df['province'].unique()
anova_freq = stats.f_oneway(*(df[df['province']==p]['claim_occurred'] for p in provinces))

# -----------------------------
# Hypothesis 2: Risk differences across zip codes
# -----------------------------
top_zipcodes = df['postalcode'].value_counts().head(10).index
anova_zip = stats.f_oneway(*(df[df['postalcode']==z]['claim_occurred'] for z in top_zipcodes))

# -----------------------------
# Hypothesis 3: Margin differences between zip codes
# -----------------------------
df['margin'] = df['totalpremium'] - df['totalclaims']
anova_margin = stats.f_oneway(*(df[df['postalcode']==z]['margin'] for z in top_zipcodes))

# -----------------------------
# Hypothesis 4: Risk differences between women and men
# -----------------------------
t_gender = stats.ttest_ind(
    df[df['gender']=='Female']['claim_occurred'],
    df[df['gender']=='Male']['claim_occurred'],
    equal_var=False
)

# -----------------------------
# Ensure reports folder exists
# -----------------------------
os.makedirs("reports", exist_ok=True)
output_file = "reports/hypothesis_results.txt"

# -----------------------------
# Save results to file
# -----------------------------
with open(output_file, "w", encoding="utf-8") as f:

    f.write("H1 - ANOVA claim frequency across provinces: " + str(anova_freq) + "\n")
    f.write("H2 - ANOVA claim frequency across top 10 zipcodes: " + str(anova_zip) + "\n")
    f.write("H3 - ANOVA margin across top 10 zipcodes: " + str(anova_margin) + "\n")
    f.write("H4 - T-test claim frequency by gender: " + str(t_gender) + "\n\n")

    f.write("Interpretation:\n")
    f.write(f"H1 - Provinces: {'Reject H0 → Statistically significant differences exist.' if anova_freq.pvalue < 0.05 else 'Fail to reject H0 → No significant differences.'}\n")
    f.write(f"H2 - Top 10 Zipcodes: {'Reject H0 → Statistically significant differences exist.' if anova_zip.pvalue < 0.05 else 'Fail to reject H0 → No significant differences.'}\n")
    f.write(f"H3 - Margin by Top 10 Zipcodes: {'Reject H0 → Statistically significant differences exist.' if anova_margin.pvalue < 0.05 else 'Fail to reject H0 → No significant differences.'}\n")
    f.write(f"H4 - Gender: {'Reject H0 → Significant difference between women and men' if t_gender.pvalue < 0.05 else 'Fail to reject H0 → No significant difference between women and men'}\n")

print(f"Results saved to {output_file}")
