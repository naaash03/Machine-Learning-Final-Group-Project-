import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load updated youth-focused dataset
df = pd.read_csv("updated_remote_work_survey_data.csv")

# Clean column names
df.columns = [col.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") for col in df.columns]

# Plot 1: Distribution of Work Arrangements
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="Current_Work_Arrangement")
plt.title("Distribution of Work Arrangements")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Plot 2: Productivity Ratings by Work Arrangement
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Current_Work_Arrangement", y="Overall_Productivity_1_5")
plt.title("Productivity by Work Arrangement")
plt.xlabel("Work Arrangement")
plt.ylabel("Productivity Rating (1-5)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Plot 3: Correlation Heatmap
numeric_df = df[[
    "Hours_Worked_Per_Week",
    "Tasks_Completed_Per_Week",
    "Overall_Productivity_1_5",
    "Work_Life_Balance_1_5",
    "Collaboration_Tool_Satisfaction_1_5",
    "Manager_Support_1_5"
]]
plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Regression Model
df['Work_Arrangement_Cat'] = df['Current_Work_Arrangement'].astype('category')
model = smf.ols(
    'Overall_Productivity_1_5 ~ C(Work_Arrangement_Cat) + Hours_Worked_Per_Week + Tasks_Completed_Per_Week + Collaboration_Tool_Satisfaction_1_5 + Manager_Support_1_5 + Work_Life_Balance_1_5',
    data=df
).fit()

# ANOVA and model summary
anova_result = sm.stats.anova_lm(model, typ=2)
model_summary = model.summary()

anova_result, model_summary.tables[1]
