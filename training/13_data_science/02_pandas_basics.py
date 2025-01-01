"""
Demonstration of Pandas fundamentals and data manipulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Data creation and basic operations
def demonstrate_dataframe_creation():
    """Examples of different ways to create DataFrames."""
    
    # From dictionary
    dict_data = {
        'name': ['John', 'Anna', 'Peter', 'Linda'],
        'age': [28, 34, 29, 32],
        'city': ['New York', 'Paris', 'Berlin', 'London'],
        'salary': [50000, 65000, 55000, 60000]
    }
    df_dict = pd.DataFrame(dict_data)
    
    # From list of dictionaries
    list_data = [
        {'date': '2023-01-01', 'value': 100},
        {'date': '2023-01-02', 'value': 110},
        {'date': '2023-01-03', 'value': 120}
    ]
    df_list = pd.DataFrame(list_data)
    
    # From NumPy array
    array_data = np.random.rand(4, 3)
    df_array = pd.DataFrame(
        array_data,
        columns=['A', 'B', 'C'],
        index=['row1', 'row2', 'row3', 'row4']
    )
    
    # Time series data
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    df_time = pd.DataFrame(
        np.random.randn(5, 3),
        index=dates,
        columns=['A', 'B', 'C']
    )
    
    return {
        'from_dict': df_dict,
        'from_list': df_list,
        'from_array': df_array,
        'time_series': df_time
    }

# Data manipulation
def demonstrate_data_manipulation():
    """Examples of data manipulation operations."""
    
    # Create sample data
    df = pd.DataFrame({
        'name': ['John', 'Anna', 'Peter', 'Linda', 'John'],
        'age': [28, 34, 29, 32, 28],
        'city': ['New York', 'Paris', 'Berlin', 'London', 'Boston'],
        'salary': [50000, 65000, 55000, 60000, 52000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'Marketing']
    })
    
    # Filtering
    high_salary = df[df['salary'] > 55000]
    it_dept = df[df['department'] == 'IT']
    
    # Grouping and aggregation
    dept_stats = df.groupby('department').agg({
        'salary': ['mean', 'min', 'max', 'count'],
        'age': ['mean', 'min', 'max']
    })
    
    # Sorting
    sorted_by_salary = df.sort_values('salary', ascending=False)
    
    # Adding/modifying columns
    df['bonus'] = df['salary'] * 0.1
    df['full_name'] = df['name'] + ' from ' + df['city']
    
    # Pivot tables
    pivot = pd.pivot_table(
        df,
        values='salary',
        index='department',
        columns='city',
        aggfunc='mean',
        fill_value=0
    )
    
    return {
        'original': df,
        'high_salary': high_salary,
        'it_dept': it_dept,
        'dept_stats': dept_stats,
        'sorted': sorted_by_salary,
        'pivot': pivot
    }

# Data cleaning
def demonstrate_data_cleaning():
    """Examples of data cleaning operations."""
    
    # Create sample data with issues
    df = pd.DataFrame({
        'name': ['John', 'Anna', None, 'Linda', 'John'],
        'age': [28, -34, 29, 32, None],
        'city': ['New York', 'paris', 'BERLIN', 'London', ''],
        'salary': ['50000', '65000', 'N/A', '60000', '52000'],
        'date': ['2023-01-01', '2023-13-01', '2023-01-01', '2023-01-01', None]
    })
    
    # Handle missing values
    df_cleaned = df.copy()
    df_cleaned['name'].fillna('Unknown', inplace=True)
    df_cleaned['age'].fillna(df_cleaned['age'].mean(), inplace=True)
    
    # Data type conversion
    df_cleaned['salary'] = pd.to_numeric(df_cleaned['salary'], errors='coerce')
    df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], errors='coerce')
    
    # String normalization
    df_cleaned['city'] = df_cleaned['city'].str.title()
    df_cleaned['city'].replace('', 'Unknown', inplace=True)
    
    # Value validation
    df_cleaned.loc[df_cleaned['age'] < 0, 'age'] = None
    
    # Remove duplicates
    df_cleaned = df_cleaned.drop_duplicates(subset=['name', 'age'])
    
    return {
        'original': df,
        'cleaned': df_cleaned,
        'info': df_cleaned.info(),
        'description': df_cleaned.describe(),
        'missing': df_cleaned.isnull().sum()
    }

# Data analysis and visualization
def demonstrate_analysis():
    """Examples of data analysis and visualization."""
    
    # Create sample time series data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(100).cumsum(),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Time series analysis
    rolling_mean = df['value'].rolling(window=7).mean()
    monthly_avg = df.resample('M', on='date')['value'].mean()
    
    # Correlation analysis
    df['value2'] = df['value'].shift(1) + np.random.randn(100) * 0.1
    correlation = df[['value', 'value2']].corr()
    
    # Category analysis
    category_stats = df.groupby('category').agg({
        'value': ['count', 'mean', 'std']
    })
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Time series plot
    plt.subplot(2, 2, 1)
    plt.plot(df['date'], df['value'], label='Original')
    plt.plot(df['date'], rolling_mean, label='7-day Rolling Mean')
    plt.title('Time Series Analysis')
    plt.legend()
    
    # Histogram
    plt.subplot(2, 2, 2)
    df['value'].hist(bins=20)
    plt.title('Value Distribution')
    
    # Box plot by category
    plt.subplot(2, 2, 3)
    df.boxplot(column='value', by='category')
    plt.title('Value by Category')
    
    # Scatter plot
    plt.subplot(2, 2, 4)
    plt.scatter(df['value'], df['value2'])
    plt.title('Value Correlation')
    
    plt.tight_layout()
    
    return {
        'data': df,
        'rolling_mean': rolling_mean,
        'monthly_avg': monthly_avg,
        'correlation': correlation,
        'category_stats': category_stats,
        'plot': plt
    }

if __name__ == '__main__':
    # DataFrame creation examples
    dataframes = demonstrate_dataframe_creation()
    print("\nDataFrame Creation Examples:")
    for name, df in dataframes.items():
        print(f"\n{name}:\n{df}")
    
    # Data manipulation examples
    manipulations = demonstrate_data_manipulation()
    print("\nData Manipulation Examples:")
    for name, result in manipulations.items():
        print(f"\n{name}:\n{result}")
    
    # Data cleaning examples
    cleaning = demonstrate_data_cleaning()
    print("\nData Cleaning Examples:")
    for name, result in cleaning.items():
        print(f"\n{name}:\n{result}")
    
    # Analysis examples
    analysis = demonstrate_analysis()
    print("\nAnalysis Examples:")
    for name, result in analysis.items():
        if name != 'plot':
            print(f"\n{name}:\n{result}")
    
    # Show plots
    plt.show() 