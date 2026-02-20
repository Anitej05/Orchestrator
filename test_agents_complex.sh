#!/bin/bash
# Complex Agent Testing Script
# Tests agents with real-world complex scenarios

BASE_URL="http://localhost:8000"
TIMEOUT=120

echo "======================================================================"
echo "  COMPREHENSIVE AGENT TESTING SUITE"
echo "  Testing with complex real-world scenarios"
echo "======================================================================"

# Helper function to run test with timeout
run_test() {
    local agent=$1
    local test_name=$2
    local prompt=$3
    local files=$4
    
    echo ""
    echo "Testing: $test_name"
    
    if [ -z "$files" ]; then
        RESULT=$(curl -s -m $TIMEOUT -X POST "$BASE_URL/api/chat" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": \"$prompt\"}" 2>&1)
    else
        RESULT=$(curl -s -m $TIMEOUT -X POST "$BASE_URL/api/chat" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": \"$prompt\", \"files\": $files}" 2>&1)
    fi
    
    if echo "$RESULT" | grep -q '"final_response"'; then
        echo "  ✅ PASS | $test_name"
        echo "  Response preview: $(echo $RESULT | grep -o '"final_response":"[^"]*"' | head -c 200)"
    elif echo "$RESULT" | grep -q '"pending_user_input"'; then
        echo "  ⚠️ NEEDS INPUT | $test_name"
        echo "  Question: $(echo $RESULT | grep -o '"question_for_user":"[^"]*"')"
    else
        echo "  ❌ FAIL | $test_name"
        echo "  Error: $(echo $RESULT | head -c 300)"
    fi
}

# ============================================================================
# SPREADSHEET AGENT - COMPLEX TESTS
# ============================================================================

echo ""
echo "======================================================================"
echo "  SPREADSHEET AGENT - COMPLEX TESTS"
echo "======================================================================"

# Create complex sales data
TEST_CSV="/tmp/complex_sales.csv"
cat > $TEST_CSV << 'EOF'
Date,Product,Category,Region,Salesperson,Quantity,Unit_Price,Total,Discount,Net_Sale,Profit_Margin
2024-01-01,Laptop Pro,Electronics,North,Alice,5,1200,6000,5,5700,15
2024-01-02,Wireless Mouse,Electronics,South,Bob,20,25,500,0,500,30
2024-01-03,Office Chair,Furniture,East,Charlie,3,350,1050,10,945,20
2024-01-04,Standing Desk,Furniture,West,Diana,2,800,1600,0,1600,25
2024-01-05,Monitor 27",Electronics,North,Alice,8,400,3200,5,3040,18
2024-01-06,Keyboard RGB,Electronics,South,Bob,15,80,1200,0,1200,35
2024-01-07,Desk Lamp,Furniture,East,Charlie,10,45,450,0,450,40
2024-01-08,Webcam HD,Electronics,West,Diana,6,120,720,5,684,22
2024-01-09,Laptop Pro,Electronics,South,Bob,3,1200,3600,5,3420,15
2024-01-10,Office Chair,Furniture,North,Alice,4,350,1400,10,1260,20
2024-01-11,Standing Desk,Furniture,East,Charlie,1,800,800,0,800,25
2024-01-12,Monitor 27",Electronics,West,Diana,5,400,2000,5,1900,18
2024-01-13,Wireless Mouse,Electronics,North,Alice,30,25,750,0,750,30
2024-01-14,Keyboard RGB,Electronics,East,Charlie,12,80,960,0,960,35
2024-01-15,Desk Lamp,Furniture,South,Bob,8,45,360,0,360,40
EOF

echo ""
echo "📊 Test 1: Multi-dimensional Analysis"
run_test "spreadsheet" "Multi-dimensional Sales Analysis" \
    "Analyze this sales data and provide: 1) Total sales by region, 2) Top performing salesperson, 3) Most profitable category, 4) Average discount percentage by product type. File: $TEST_CSV" \
    "[{\"file_name\": \"complex_sales.csv\", \"file_path\": \"$TEST_CSV\", \"file_type\": \"spreadsheet\"}]"

echo ""
echo "📊 Test 2: Trend Analysis"
run_test "spreadsheet" "Sales Trend Analysis" \
    "Looking at this sales data, identify any trends in sales over time and suggest which products are gaining momentum. File: $TEST_CSV" \
    "[{\"file_name\": \"complex_sales.csv\", \"file_path\": \"$TEST_CSV\", \"file_type\": \"spreadsheet\"}]"

# ============================================================================
# DOCUMENT AGENT - COMPLEX TESTS
# ============================================================================

echo ""
echo "======================================================================"
echo "  DOCUMENT AGENT - COMPLEX TESTS"
echo "======================================================================"

# Create complex business document
TEST_DOC="/tmp/business_report.txt"
cat > $TEST_DOC << 'EOF'
STRATEGIC BUSINESS REVIEW - FY 2024

EXECUTIVE SUMMARY
This comprehensive review analyzes our company's performance across all business units 
for the fiscal year 2024. Total revenue reached $45.2 million, representing a 23% 
year-over-year growth. Our market share increased from 12% to 15% in the primary 
market segment.

FINANCIAL PERFORMANCE
Q1 Revenue: $9.8M (Operating Margin: 18%)
Q2 Revenue: $10.5M (Operating Margin: 20%)
Q3 Revenue: $11.8M (Operating Margin: 22%)
Q4 Revenue: $13.1M (Operating Margin: 24%)

Key Financial Metrics:
- Gross Revenue: $45.2M
- Operating Expenses: $32.4M
- Net Profit: $12.8M
- EBITDA: $15.2M
- Cash Reserves: $8.5M

MARKET ANALYSIS
Primary Markets:
1. North America (45% of revenue) - Growth: +18%
2. Europe (30% of revenue) - Growth: +28%
3. Asia Pacific (25% of revenue) - Growth: +35%

Competitive Position:
- Market Leader: TechCorp Inc. (28% market share)
- Our Position: #3 (15% market share)
- Key Differentiator: AI-powered automation features

STRATEGIC INITIATIVES
1. Product Innovation: Launched 3 new product lines, contributing $8.2M in revenue
2. Geographic Expansion: Entered 4 new markets in APAC region
3. Customer Success: Achieved 94% customer retention rate
4. Operational Efficiency: Reduced operational costs by 12% through automation

RISKS AND CHALLENGES
- Supply chain disruptions affected Q2 deliveries (estimated impact: $1.2M)
- Increased competition in European market
- Talent acquisition challenges in engineering roles
- Regulatory changes in data privacy affecting product roadmap

OUTLOOK FOR FY 2025
Projected Revenue: $58M (28% growth)
Key Focus Areas:
1. AI/ML product enhancement
2. Strategic partnerships with cloud providers
3. Expansion into Latin American markets
4. Investment in customer success infrastructure

RECOMMENDATIONS
1. Increase R&D budget by 25% to accelerate AI product development
2. Establish regional headquarters in Singapore for APAC growth
3. Implement advanced analytics for customer behavior prediction
4. Develop strategic partnership program with system integrators
EOF

echo ""
echo "📄 Test 1: Comprehensive Document Analysis"
run_test "document" "Business Report Analysis" \
    "Analyze this business report and extract: 1) Key financial metrics with values, 2) Main strategic initiatives and their outcomes, 3) Top 3 risks with mitigation suggestions, 4) Summary of FY 2025 outlook. File: $TEST_DOC" \
    "[{\"file_name\": \"business_report.txt\", \"file_path\": \"$TEST_DOC\", \"file_type\": \"document\"}]"

echo ""
echo "📄 Test 2: Comparative Analysis"
run_test "document" "Quarterly Performance Comparison" \
    "Compare the quarterly performance in this report. Which quarter performed best and why? What factors contributed to the improvement throughout the year? File: $TEST_DOC" \
    "[{\"file_name\": \"business_report.txt\", \"file_path\": \"$TEST_DOC\", \"file_type\": \"document\"}]"

# ============================================================================
# MAIL AGENT - COMPLEX TESTS
# ============================================================================

echo ""
echo "======================================================================"
echo "  MAIL AGENT - COMPLEX TESTS"
echo "======================================================================"

echo ""
echo "📧 Test 1: Complex Email Search and Analysis"
run_test "mail" "Email Search with Multiple Criteria" \
    "Search my emails for messages about project updates or meeting schedules from the last 2 weeks, then summarize the key action items mentioned in those emails"

echo ""
echo "📧 Test 2: Email Draft Request"
run_test "mail" "Draft Professional Email" \
    "Draft a professional email to my team summarizing the Q4 project milestones and requesting status updates for the upcoming deadline"

# ============================================================================
# UNIVERSAL AGENT - COMPLEX TESTS
# ============================================================================

echo ""
echo "======================================================================"
echo "  UNIVERSAL AGENT - COMPLEX TESTS"
echo "======================================================================"

echo ""
echo "🤖 Test 1: Code Generation Task"
run_test "universal" "Python Code Generation" \
    "Write a Python function that calculates the Fibonacci sequence up to n terms, with proper error handling and documentation. Include example usage."

echo ""
echo "🤖 Test 2: Data Analysis Task"
run_test "universal" "Statistical Analysis" \
    "Explain how to perform a linear regression analysis on a dataset. Include the mathematical formula, assumptions, and a step-by-step guide for implementation in Python."

echo ""
echo "🤖 Test 3: Research and Synthesis"
run_test "universal" "Technology Research" \
    "Research and compare REST API vs GraphQL. Provide: 1) Key differences, 2) Use cases for each, 3) Performance considerations, 4) Recommendation for a microservices architecture"

echo ""
echo "🤖 Test 4: Problem Solving"
run_test "universal" "Algorithm Design" \
    "Design an algorithm to find the longest common subsequence between two strings. Explain the approach, provide pseudocode, and analyze the time complexity."

echo ""
echo "🤖 Test 5: File System Task"
run_test "universal" "Directory Analysis" \
    "List all Python files in the /home/clawuser/Orchestrator/backend directory and count how many contain the word 'agent' in their filename"

# ============================================================================
# ZOHO BOOKS AGENT - COMPLEX TESTS
# ============================================================================

echo ""
echo "======================================================================"
echo "  ZOHO BOOKS AGENT - COMPLEX TESTS"
echo "======================================================================"

echo ""
echo "💰 Test 1: Invoice Analysis"
run_test "zoho_books" "Invoice Status Query" \
    "Show me all unpaid invoices from the last 30 days, grouped by customer, with the total outstanding amount"

echo ""
echo "💰 Test 2: Financial Report Request"
run_test "zoho_books" "Revenue Summary" \
    "Generate a summary of total revenue by customer for the current quarter, highlighting the top 5 customers by revenue"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "======================================================================"
echo "  TEST SUMMARY"
echo "======================================================================"
echo ""
echo "  Complex tests completed for all agents."
echo ""
echo "  Agents Tested:"
echo "    - Spreadsheet Agent: Multi-dimensional analysis, Trend analysis"
echo "    - Document Agent: Business report analysis, Quarterly comparison"
echo "    - Mail Agent: Complex search, Email drafting"
echo "    - Universal Agent: Code gen, Analysis, Research, Algorithms, File system"
echo "    - Zoho Books Agent: Invoice queries, Financial reports"
echo ""
echo "  Note: Results depend on:"
echo "    - LLM availability and response time"
echo "    - API credentials (Mail, Zoho Books)"
echo "    - File system access (Universal Agent)"
echo ""
echo "======================================================================"

# Cleanup
rm -f $TEST_CSV $TEST_DOC