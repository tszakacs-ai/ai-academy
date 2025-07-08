import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('adult.data')
df.columns = ['age', 'workclass', 'fnlwgt', 'education','education-num', 'marital-status', 'occupation', 'relationship','race',
              'sex','capital-gain','capital-loss','hours-per-week','native-country','salary']

# Show basic info
print(df.info())
print(df.head())


# Show distribution of categorical variables
categorical_vars = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
for var in categorical_vars:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=var, order=df[var].value_counts().index)
    plt.title(f'Distribution of {var}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Salary distribution by race
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='race', hue='salary')
plt.title('Salary Distribution by Race')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Salary distribution by gender
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='sex', hue='salary')
plt.title('Salary Distribution by Gender')
plt.tight_layout()
plt.show()

# Salary distribution by race and gender
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='race', hue='sex')
plt.title('Race and Gender Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cross-tabulation of salary by race and gender
print(pd.crosstab([df['race'], df['sex']], df['salary'], normalize='index'))
# ...existing code...