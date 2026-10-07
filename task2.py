# ============================================================
# CODSOFT - DATA ANALYTICS INTERNSHIP
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import numpy as np

print("=" * 70)
print("CODSOFT - DATA ANALYTICS INTERNSHIP")
print("TASK 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ============================================================
# 1. CREATE DATASET
# ============================================================

data = {
    "PassengerId": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20
    ],

    "Name": [
        "John", "Sarah", "Ravi", "Anita", "David",
        "Priya", "Arun", "Meena", "John", "Rahul",
        "Swathi", "Kiran", "Latha", "Manoj", "Neha",
        "Varun", "Pooja", "Ramesh", "Divya", "Ajay"
    ],

    "Age": [
        22, 28, 25, 28, 35,
        29, 31, 27, 22, 28,
        26, 30, 24, 28, 32,
        29, 27, 33, 25, 28
    ],

    "Gender": [
        "Male", "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Male",
        "Female", "Male", "Female", "Male", "Female",
        "Male", "Female", "Male", "Female", "Male"
    ],

    "City": [
        "Hyderabad", "Hyderabad", "Chennai", "Bangalore", "Mumbai",
        "Delhi", "Hyderabad", "Chennai", "Hyderabad", "Mumbai",
        "Bangalore", "Hyderabad", "Delhi", "Chennai", "Mumbai",
        "Hyderabad", "Bangalore", "Delhi", "Chennai", "Hyderabad"
    ],

    "Salary": [
        25000, 32000, 28000, 35000, 40000,
        35000, 45000, 30000, 25000, 38000,
        36000, 42000, 29000, 31000, 50000,
        41000, 34000, 39000, 30000, 37000
    ],

    "Department": [
        "IT", "HR", "IT", "Finance", "Sales",
        "HR", "IT", "Finance", "IT", "Sales",
        "HR", "IT", "Finance", "Sales", "HR",
        "IT", "Finance", "Sales", "HR", "IT"
    ]
}

df = pd.DataFrame(data)


# ============================================================
# 2. DISPLAY DATASET
# ============================================================

print("\n1. ORIGINAL DATASET")
print("-" * 70)
print(df.to_string(index=False))


# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print("\n2. DATASET INFORMATION")
print("-" * 70)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(list(df.columns))


# ============================================================
# 4. DATA TYPES
# ============================================================

print("\n3. DATA TYPES")
print("-" * 70)
print(df.dtypes)


# ============================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================

print("\n4. DESCRIPTIVE STATISTICS")
print("-" * 70)

print(df.describe())


# ============================================================
# 6. AGE ANALYSIS
# ============================================================

print("\n5. AGE ANALYSIS")
print("-" * 70)

print("Mean Age:", round(df["Age"].mean(), 2))
print("Median Age:", df["Age"].median())
print("Minimum Age:", df["Age"].min())
print("Maximum Age:", df["Age"].max())


# ============================================================
# 7. SALARY ANALYSIS
# ============================================================

print("\n6. SALARY ANALYSIS")
print("-" * 70)

print("Average Salary:", round(df["Salary"].mean(), 2))
print("Median Salary:", df["Salary"].median())
print("Minimum Salary:", df["Salary"].min())
print("Maximum Salary:", df["Salary"].max())


# ============================================================
# 8. GENDER DISTRIBUTION
# ============================================================

print("\n7. GENDER DISTRIBUTION")
print("-" * 70)

print(df["Gender"].value_counts())


# ============================================================
# 9. CITY DISTRIBUTION
# ============================================================

print("\n8. CITY DISTRIBUTION")
print("-" * 70)

print(df["City"].value_counts())


# ============================================================
# 10. DEPARTMENT DISTRIBUTION
# ============================================================

print("\n9. DEPARTMENT DISTRIBUTION")
print("-" * 70)

print(df["Department"].value_counts())


# ============================================================
# 11. AVERAGE SALARY BY DEPARTMENT
# ============================================================

print("\n10. AVERAGE SALARY BY DEPARTMENT")
print("-" * 70)

department_salary = df.groupby("Department")["Salary"].mean().sort_values(
    ascending=False
)

print(department_salary)


# ============================================================
# 12. AVERAGE SALARY BY CITY
# ============================================================

print("\n11. AVERAGE SALARY BY CITY")
print("-" * 70)

city_salary = df.groupby("City")["Salary"].mean().sort_values(
    ascending=False
)

print(city_salary)


# ============================================================
# 13. AVERAGE SALARY BY GENDER
# ============================================================

print("\n12. AVERAGE SALARY BY GENDER")
print("-" * 70)

gender_salary = df.groupby("Gender")["Salary"].mean()

print(gender_salary)


# ============================================================
# 14. AVERAGE AGE BY DEPARTMENT
# ============================================================

print("\n13. AVERAGE AGE BY DEPARTMENT")
print("-" * 70)

department_age = df.groupby("Department")["Age"].mean()

print(department_age)


# ============================================================
# 15. HIGHEST SALARY
# ============================================================

print("\n14. HIGHEST SALARY RECORD")
print("-" * 70)

highest_salary = df.loc[df["Salary"].idxmax()]

print(highest_salary.to_string())


# ============================================================
# 16. LOWEST SALARY
# ============================================================

print("\n15. LOWEST SALARY RECORD")
print("-" * 70)

lowest_salary = df.loc[df["Salary"].idxmin()]

print(lowest_salary.to_string())


# ============================================================
# 17. SALARY RANGE
# ============================================================

print("\n16. SALARY RANGE")
print("-" * 70)

salary_range = df["Salary"].max() - df["Salary"].min()

print("Salary Range:", salary_range)


# ============================================================
# 18. AGE-SALARY CORRELATION
# ============================================================

print("\n17. AGE AND SALARY RELATIONSHIP")
print("-" * 70)

correlation = df["Age"].corr(df["Salary"])

print("Correlation between Age and Salary:",
      round(correlation, 3))


# ============================================================
# 19. OUTLIER DETECTION USING IQR
# ============================================================

print("\n18. OUTLIER DETECTION")
print("-" * 70)

Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

outliers = df[
    (df["Salary"] < lower_limit) |
    (df["Salary"] > upper_limit)
]

print("\nSalary Outliers:")
if len(outliers) == 0:
    print("No salary outliers detected.")
else:
    print(outliers.to_string(index=False))


# ============================================================
# 20. DEPARTMENT WITH HIGHEST AVERAGE SALARY
# ============================================================

print("\n19. HIGHEST PAYING DEPARTMENT")
print("-" * 70)

highest_department = department_salary.idxmax()
highest_department_salary = department_salary.max()

print("Department:", highest_department)
print("Average Salary:", round(highest_department_salary, 2))


# ============================================================
# 21. CITY WITH HIGHEST AVERAGE SALARY
# ============================================================

print("\n20. CITY WITH HIGHEST AVERAGE SALARY")
print("-" * 70)

highest_city = city_salary.idxmax()
highest_city_salary = city_salary.max()

print("City:", highest_city)
print("Average Salary:", round(highest_city_salary, 2))


# ============================================================
# 22. BUSINESS INSIGHTS
# ============================================================

print("\n21. KEY BUSINESS INSIGHTS")
print("-" * 70)

print("1. The dataset contains", len(df), "records.")
print("2. The average salary is", round(df["Salary"].mean(), 2))
print("3. The average age is", round(df["Age"].mean(), 2))
print("4. The highest average salary department is",
      highest_department)
print("5. The city with the highest average salary is",
      highest_city)
print("6. The correlation between Age and Salary is",
      round(correlation, 3))

if len(outliers) == 0:
    print("7. No significant salary outliers were detected.")
else:
    print("7. Salary outliers were detected.")


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n22. FINAL EDA SUMMARY")
print("-" * 70)

print("Total Records:", df.shape[0])
print("Total Columns:", df.shape[1])
print("Average Age:", round(df["Age"].mean(), 2))
print("Average Salary:", round(df["Salary"].mean(), 2))
print("Median Salary:", df["Salary"].median())
print("Minimum Salary:", df["Salary"].min())
print("Maximum Salary:", df["Salary"].max())
print("Salary Range:", salary_range)
print("Salary Outliers:", len(outliers))


# ============================================================
# 24. SAVE RESULTS
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Total Records",
        "Total Columns",
        "Average Age",
        "Average Salary",
        "Median Salary",
        "Minimum Salary",
        "Maximum Salary",
        "Salary Range",
        "Age-Salary Correlation",
        "Salary Outliers"
    ],

    "Value": [
        df.shape[0],
        df.shape[1],
        round(df["Age"].mean(), 2),
        round(df["Salary"].mean(), 2),
        df["Salary"].median(),
        df["Salary"].min(),
        df["Salary"].max(),
        salary_range,
        round(correlation, 3),
        len(outliers)
    ]
})

summary.to_csv("task2_eda_summary.csv", index=False)

print("\n23. FILE SAVED")
print("-" * 70)
print("EDA summary saved successfully!")
print("File name: task2_eda_summary.csv")


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("TASK 2 - EDA COMPLETED SUCCESSFULLY!")
print("=" * 70)