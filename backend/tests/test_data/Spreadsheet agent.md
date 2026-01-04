RETAIL\_SALES\_QUERIES = {   
    "simple": [   
        \# Basic info   
        "How many rows are in this dataset?",   
        "What columns does this dataset have?",   
        "How many transactions are there in total?",   
           
        \# Simple aggregations   
        "What is the total Total Amount?",   
        "What is the maximum Total Amount?",   
        "What is the minimum Total Amount?",   
        "What is the average Total Amount?",   
           
        \# Count operations   
        "How many unique Customer ID are there?",   
        "How many unique Product Category are there?",   
        "How many Male customers are there?",   
        "How many Female customers are there?",   
           
        \# Basic statistics   
        "What is the average Age of customers?",   
        "What is the average Quantity purchased?",   
        "What is the highest Price per Unit?",   
    ],   
       
    "medium": [   
        \# Group by \+ aggregation   
        "What is the total Total Amount for each Product Category?",   
        "What is the average Total Amount by Gender?",   
        "What is the total Total Amount by Product Category?",   
        "What is the average Age by Gender?",   
        "What is the total Quantity sold for each Product Category?",   
           
        \# Ranking queries   
        "Which Product Category has the highest total sales?",   
        "Which Gender spends more on average?",   
        "Show the top 5 transactions by Total Amount",   
        "Show the top 3 Product Category by total revenue",   
           
        \# Filtering \+ aggregation   
        "How many transactions have Total Amount greater than 50?",   
        "What is the average Total Amount for customers over 30 years old?",   
        "How many transactions are for Beauty products?",   
        "What is the total sales for Electronics category?",   
           
        \# Date\-based queries   
        "What is the total Total Amount by month?",   
        "How many transactions happened in January 2023?",   
           
        \# Multiple column grouping   
        "What is the average Total Amount by Product Category and Gender?",   
    ],   
       
    "hard": [   
        \# Percentage calculations   
        "What percentage of total sales does each Product Category represent?",   
        "What percentage of customers are Male vs Female?",   
           
        \# Multi\-level aggregation   
        "For each Product Category, show total sales, average price, and transaction count",   
        "Calculate total revenue, average transaction value, and customer count by Gender",   
           
        \# Complex filtering   
        "Which customers have Total Amount above the average and made more than 2 purchases?",   
        "Show Product Categories where average Total Amount is greater than 100",   
           
        \# Statistical analysis   
        "What is the standard deviation of Total Amount?",   
        "Calculate the correlation between Age and Total Amount",   
           
        \# Temporal analysis   
        "Calculate month-over-month growth in Total Amount",   
        "Which month had the highest total sales?",   
           
        \# Multi\-criteria analysis   
        "Show the average Total Amount by Product Category for customers over 25 years old",   
        "Which Gender spends more in the Electronics category?",   
           
        \# Edge cases   
        "Are there any missing values in the Total Amount column?",   
        "How many duplicate Customer ID entries exist?",   
    ]   
}   
\# ============================================================================   
\# ZARA SALES DATASET \(254 rows\)   
\# Columns: Product ID, Product Position, Promotion, Product Category,    
\#          Seasonal, Sales Volume, brand, price, section, terms   
\# ============================================================================   
ZARA\_SALES\_QUERIES = {   
    "simple": [   
        \# Basic info   
        "How many rows are in this dataset?",   
        "How many products are in this dataset?",   
        "What columns are available?",   
           
        \# Simple aggregations   
        "What is the total Sales Volume?",   
        "What is the average price?",   
        "What is the maximum price?",   
        "What is the minimum price?",   
           
        \# Count operations   
        "How many unique section are there?",   
        "How many unique Product Position are there?",   
        "How many products have Promotion as Yes?",   
        "How many products have Promotion as No?",   
        "How many products are marked as Seasonal?",   
           
        \# Basic categorical counts   
        "How many products are in the MAN section?",   
        "How many products are in the WOMAN section?",   
        "How many products are displayed at End-cap position?",   
    ],   
       
    "medium": [   
        \# Group by \+ aggregation   
        "What is the total Sales Volume for each section?",   
        "What is the average price by Product Position?",   
        "What is the total Sales Volume by Promotion status?",   
        "What is the average price for each section?",   
           
        \# Ranking queries   
        "Which section has the highest total Sales Volume?",   
        "Which Product Position has the highest average price?",   
        "Show the top 10 products by Sales Volume",   
        "Show the top 5 most expensive products",   
           
        \# Filtering \+ aggregation   
        "What is the average price for products on Promotion?",   
        "How many products have Sales Volume greater than 100?",   
        "What is the total Sales Volume for Seasonal products?",   
        "How many products are priced above 50 dollars?",   
           
        \# Comparison queries   
        "What is the average price difference between promoted and non-promoted products?",   
        "Compare average Sales Volume between Aisle and End-cap positions",   
           
        \# Multi\-column grouping   
        "What is the total Sales Volume by section and Product Position?",   
        "What is the average price by section and Promotion status?",   
    ],   
       
    "hard": [   
        \# Percentage calculations   
        "What percentage of total Sales Volume does each section represent?",   
        "What percentage of products are on Promotion?",   
        "What is the promotion rate by section?",   
           
        \# Multi\-level aggregation   
        "For each section, show total Sales Volume, average price, and product count",   
        "Calculate total sales, average price, and promotion rate by Product Position",   
           
        \# Complex filtering   
        "Which sections have above-average Sales Volume and more than 20 products?",   
        "Show products where price is above average AND Sales Volume is above average",   
           
        \# Statistical analysis   
        "What is the standard deviation of price by section?",   
        "Calculate the price range (max \- min\) for each section?",   
           
        \# Multi\-criteria analysis   
        "What is the average price for Seasonal products on Promotion by section?",   
        "Compare Sales Volume between MAN and WOMAN sections for promoted products only",   
        "Which Product Position generates the most Sales Volume for non-Seasonal items?",   
           
        \# Promotion effectiveness   
        "Calculate the average Sales Volume uplift for promoted vs non-promoted products",   
        "Which section benefits most from promotions in terms of Sales Volume?",   
           
        \# Edge cases   
        "Are there any products with zero Sales Volume?",   
        "How many products have missing price information?",   
        "Identify products with the same price but different Sales Volume",   
    ]   
}   
\# ============================================================================   
\# FINANCIALS DATASET \(353 rows\)   
\# Columns: Account, Business Unit, Currency, Year, Scenario,   
\#          Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec   
\# ============================================================================   
FINANCIALS_QUERIES = {   
    "simple": [   
        \# Basic info   
        "How many rows are in this dataset?",   
        "How many financial records are there?",   
        "What columns does this dataset have?",   
           
        \# Simple aggregations   
        "What is the total value in January?",   
        "What is the total value in December?",   
        "What is the maximum value in any month?",   
        "What is the average value across all months?",   
           
        \# Count operations   
        "How many unique Account types are there?",   
        "How many unique Business Unit are there?",   
        "How many unique Year values are there?",   
        "How many records are for Scenario Actuals?",   
        "How many records are for Scenario Forecast?",   
           
        \# Basic categorical queries   
        "List all unique Account types",   
        "How many records are for the year 2012?",   
    ],   
       
    "medium": [   
        \# Group by \+ aggregation   
        "What is the total value across all months for each Account?",   
        "What is the average January value by Business Unit?",   
        "What is the total annual value by Year?",   
        "What is the total value by Scenario?",   
           
        \# Ranking queries   
        "Which Account has the highest total across all months?",   
        "Which Business Unit has the highest total revenue?",   
        "Which month has the highest average value across all accounts?",   
           
        \# Quarterly analysis   
        "What is the total for Q1 (Jan, Feb, Mar)?",   
        "What is the total for Q2 (Apr, May, Jun)?",   
        "What is the total for Q3 (Jul, Aug, Sep)?",   
        "What is the total for Q4 (Oct, Nov, Dec)?",   
           
        \# Filtering \+ aggregation   
        "What is the total value for Sales account only?",   
        "What is the total Cost of Goods Sold across all months?",   
        "Show the total values for all Expense accounts",   
           
        \# Time\-based comparisons   
        "What is the total for the first half (Jan-Jun) vs second half (Jul-Dec)?",   
        "Compare Q1 and Q4 totals",   
           
        \# Multi\-column grouping   
        "What is the total value by Account and Business Unit?",   
        "What is the average monthly value by Year and Scenario?",   
    ],   
       
    "hard": [   
        \# Percentage calculations   
        "What percentage of total annual value does each Account represent?",   
        "What is the percentage contribution of each month to the annual total?",   
           
        \# Multi\-level aggregation   
        "For each Account, calculate total value, average monthly value, and standard deviation",   
        "Show total revenue, total costs, and profit margin by Business Unit",   
           
        \# Trend analysis   
        "Calculate the month-over-month growth rate",   
        "Which months show positive growth compared to the previous month?",   
        "Identify the month with the highest growth rate",   
           
        \# Complex filtering   
        "Show Accounts where total annual value exceeds 1 million",   
        "Which Business Units have higher Actuals than Forecast?",   
           
        \# Scenario comparison   
        "Compare Actuals vs Forecast by Account",   
        "Calculate the variance between Budget and Actuals for each month",   
        "Which Accounts have the largest Budget vs Actuals variance?",   
           
        \# Statistical analysis   
        "Calculate the coefficient of variation (std/mean) for each Account",   
        "Which Account has the most consistent monthly values?",   
           
        \# Multi\-criteria analysis   
        "For Sales accounts only, compare Actuals vs Forecast by Business Unit",   
        "Calculate average quarterly values by Account and Scenario",   
           
        \# Year\-over\-year analysis   
        "Compare total values between 2012 and 2013 by Account",   
        "Which Accounts showed growth from 2012 to 2013?",   
           
        \# Edge cases   
        "Are there any months with negative values?",   
        "Identify any missing monthly values",   
        "Which Account-Business Unit combinations have zero values across all months?",   
    ]   
}   
\# ============================================================================   
\# SALES 10K DATASET \(10,000 rows\)   
\# Columns: Region, Country, Item Type, Sales Channel, Order Priority,   
\#          Order Date, Order ID, Ship Date, Units Sold, Unit Price,    
\#          Unit Cost, Total Revenue, Total Cost, Total Profit   
\# ============================================================================   
SALES\_10K\_QUERIES = {   
    "simple": [   
        \# Basic info   
        "How many rows are in this dataset?",   
        "How many sales records are there?",   
        "What columns are available in this dataset?",   
           
        \# Simple aggregations   
        "What is the total Total Revenue?",   
        "What is the total Total Profit?",   
        "What is the total Total Cost?",   
        "What is the maximum Total Revenue?",   
        "What is the minimum Total Profit?",   
        "What is the average Unit Price?",   
        "What is the average Units Sold?",   
           
        \# Count operations   
        "How many unique Region are there?",   
        "How many unique Country are there?",   
        "How many unique Item Type are there?",   
        "How many orders were placed Online?",   
        "How many orders were placed Offline?",   
        "How many unique Sales Channel are there?",   
           
        \# Basic categorical counts   
        "How many orders have priority L?",   
        "How many orders have priority C?",   
        "How many orders are from Europe region?",   
    ],   
       
    "medium": [   
        \# Group by \+ aggregation   
        "What is the total Total Revenue for each Region?",   
        "What is the total Total Profit by Item Type?",   
        "What is the average Total Revenue by Sales Channel?",   
        "What is the total Units Sold by Region?",   
        "What is the average Unit Price by Item Type?",   
           
        \# Ranking queries   
        "Which Region has the highest total Total Revenue?",   
        "Which Country has the highest total Total Profit?",   
        "Which Item Type generates the most Total Revenue?",   
        "Show the top 10 countries by Total Profit",   
        "Show the top 5 Item Types by Units Sold",   
           
        \# Filtering \+ aggregation   
        "What is the total Total Revenue for Online sales only?",   
        "What is the average Total Profit for orders with priority H?",   
        "How many orders have Total Revenue greater than 10000?",   
        "What is the total Total Profit for Office Supplies?",   
        "What is the average Total Revenue for Offline channel?",   
           
        \# Comparison queries   
        "Compare total Total Revenue between Online and Offline channels",   
        "What is the average Total Profit difference between priority levels?",   
        "Compare average Unit Price across different Sales Channels",   
           
        \# Date\-based queries   
        "What is the total Total Revenue by month?",   
        "How many orders were placed in 2012?",   
        "What is the average Total Profit by quarter?",   
           
        \# Multi\-column grouping   
        "What is the total Total Revenue by Region and Sales Channel?",   
        "What is the average Total Profit by Item Type and Order Priority?",   
        "What is the total Units Sold by Region and Item Type?",   
    ],   
       
    "hard": [   
        \# Profit margin calculations   
        "Calculate the profit margin (Total Profit / Total Revenue) for each Item Type",   
        "Which Item Type has the highest profit margin?",   
        "What is the average profit margin by Region?",   
        "Calculate profit margin by Sales Channel and compare Online vs Offline",   
           
        \# Percentage calculations   
        "What percentage of total Total Revenue does each Region represent?",   
        "What percentage of orders are placed Online vs Offline?",   
        "What is the revenue distribution across different Item Types?",   
           
        \# Multi\-level aggregation   
        "For each Region, show total Total Revenue, total Total Profit, average profit margin, and order count",   
        "Calculate total revenue, total cost, total profit, and units sold by Item Type",   
        "Show total sales, average order value, and profit margin by Sales Channel",   
           
        \# Complex filtering   
        "Which Countries have Total Revenue above 1 million and profit margin above 20%?",   
        "Show Item Types where average Total Profit exceeds 5000 and Units Sold is above 100",   
        "Identify Regions with Total Revenue above the global average",   
           
        \# Statistical analysis   
        "What is the standard deviation of Total Profit by Region?",   
        "Calculate the coefficient of variation for Total Revenue by Item Type",   
        "Which Item Type has the most consistent profit margins?",   
           
        \# Temporal analysis   
        "Calculate month-over-month growth in Total Revenue",   
        "Which quarter had the highest total Total Profit?",   
        "Identify seasonal trends in Units Sold by Item Type",   
        "Compare Year-over-Year Total Revenue growth",   
           
        \# Multi\-criteria analysis   
        "What is the average Total Profit for Online Beverages sales in Europe?",   
        "Compare profit margins between High and Low priority orders by Region",   
        "Which Sales Channel is most profitable for each Item Type?",   
        "Show the top 3 Countries by Total Revenue in each Region",   
           
        \# Cost efficiency analysis   
        "Calculate the cost-to-revenue ratio by Item Type",   
        "Which Region has the lowest average Unit Cost?",   
        "Compare Unit Price vs Unit Cost margins across Sales Channels",   
           
        \# Order fulfillment analysis   
        "Calculate average shipping time (Ship Date \- Order Date\) by Region",   
        "Which Order Priority level has the fastest average shipping time?",   
           
        \# Edge cases and data quality   
        "Are there any orders with negative Total Profit?",   
        "Identify orders where Total Cost exceeds Total Revenue",   
        "How many orders have missing Ship Date?",   
        "Find duplicate Order ID entries",   
        "Are there any outliers in Total Revenue (values beyond 3 standard deviations)?",   
    ]   
}   
  
