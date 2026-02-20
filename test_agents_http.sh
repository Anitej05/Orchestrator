#!/bin/bash
# HTTP-based Agent Testing Script
# Tests agents through the orchestrator API

BASE_URL="http://localhost:8000"
RESULTS_FILE="agent_test_results.json"

echo "======================================================================"
echo "  ORCHESTRATOR AGENT TESTING SUITE (HTTP)"
echo "======================================================================"

# Initialize results
echo '{"timestamp": "'$(date -Iseconds)'", "results": {}}' > $RESULTS_FILE

# Helper function to print test result
print_result() {
    local agent=$1
    local test=$2
    local success=$3
    local message=$4
    
    if [ "$success" = "true" ]; then
        echo "  ✅ PASS | $test"
    else
        echo "  ❌ FAIL | $test: $message"
    fi
}

# ============================================================================
# SPREADSHEET AGENT TESTS
# ============================================================================

echo ""
echo "======================================================================"
echo "  SPREADSHEET AGENT TESTS"
echo "======================================================================"

# Create test CSV
TEST_CSV="/tmp/test_sales.csv"
cat > $TEST_CSV << 'EOF'
Product,Region,Sales,Quantity,Date
Laptop,North,1500,5,2024-01-15
Mouse,South,200,20,2024-01-16
Keyboard,North,300,15,2024-01-17
Monitor,East,800,8,2024-01-18
Laptop,South,1500,3,2024-01-19
EOF

echo ""
echo "📊 Test 1: Sales Data Analysis"
RESULT=$(curl -s -X POST "$BASE_URL/api/chat" \
    -H "Content-Type: application/json" \
    -d "{
        \"prompt\": \"Analyze this CSV file and tell me the total sales by region: $TEST_CSV\",
        \"files\": [{\"file_name\": \"test_sales.csv\", \"file_path\": \"$TEST_CSV\", \"file_type\": \"spreadsheet\"}]
    }")

if echo "$RESULT" | grep -q '"final_response"'; then
    print_result "spreadsheet" "Sales Data Analysis" "true" ""
else
    print_result "spreadsheet" "Sales Data Analysis" "false" "$(echo $RESULT | head -c 100)"
fi

# ============================================================================
# DOCUMENT AGENT TESTS
# ============================================================================

echo ""
echo "======================================================================"
echo "  DOCUMENT AGENT TESTS"
echo "======================================================================"

# Create test document
TEST_DOC="/tmp/test_report.txt"
cat > $TEST_DOC << 'EOF'
QUARTERLY BUSINESS REPORT - Q4 2024

Executive Summary:
The company achieved significant growth in Q4 2024, with total revenue reaching $2.5 million,
representing a 15% increase from the previous quarter.

Financial Highlights:
- Revenue: $2,500,000
- Operating Expenses: $1,800,000
- Net Profit: $700,000
- Profit Margin: 28%
EOF

echo ""
echo "📄 Test 1: Document Analysis"
RESULT=$(curl -s -X POST "$BASE_URL/api/chat" \
    -H "Content-Type: application/json" \
    -d "{
        \"prompt\": \"Summarize the key financial highlights from this document: $TEST_DOC\",
        \"files\": [{\"file_name\": \"test_report.txt\", \"file_path\": \"$TEST_DOC\", \"file_type\": \"document\"}]
    }")

if echo "$RESULT" | grep -q '"final_response"'; then
    print_result "document" "Document Analysis" "true" ""
else
    print_result "document" "Document Analysis" "false" "$(echo $RESULT | head -c 100)"
fi

# ============================================================================
# MAIL AGENT TESTS
# ============================================================================

echo ""
echo "======================================================================"
echo "  MAIL AGENT TESTS"
echo "======================================================================"

echo ""
echo "📧 Test 1: Email Search Request"
RESULT=$(curl -s -X POST "$BASE_URL/api/chat" \
    -H "Content-Type: application/json" \
    -d "{
        \"prompt\": \"Search my emails for messages about project updates from last week\"
    }")

if echo "$RESULT" | grep -q '"final_response"\|"pending_user_input"\|"task_agent_pairs"'; then
    print_result "mail" "Email Search Request" "true" ""
else
    print_result "mail" "Email Search Request" "false" "$(echo $RESULT | head -c 100)"
fi

# ============================================================================
# UNIVERSAL AGENT TESTS
# ============================================================================

echo ""
echo "======================================================================"
echo "  UNIVERSAL AGENT TESTS"
echo "======================================================================"

echo ""
echo "🤖 Test 1: General Reasoning Task"
RESULT=$(curl -s -X POST "$BASE_URL/api/chat" \
    -H "Content-Type: application/json" \
    -d "{
        \"prompt\": \"Explain the difference between supervised and unsupervised machine learning in simple terms\"
    }")

if echo "$RESULT" | grep -q '"final_response"'; then
    print_result "universal" "General Reasoning" "true" ""
else
    print_result "universal" "General Reasoning" "false" "$(echo $RESULT | head -c 100)"
fi

echo ""
echo "🤖 Test 2: Problem Solving Task"
RESULT=$(curl -s -X POST "$BASE_URL/api/chat" \
    -H "Content-Type: application/json" \
    -d "{
        \"prompt\": \"If a train travels at 60 mph and needs to cover 180 miles, how long will it take?\"
    }")

if echo "$RESULT" | grep -q '"final_response"'; then
    print_result "universal" "Problem Solving" "true" ""
else
    print_result "universal" "Problem Solving" "false" "$(echo $RESULT | head -c 100)"
fi

# ============================================================================
# ZOHO BOOKS AGENT TESTS
# ============================================================================

echo ""
echo "======================================================================"
echo "  ZOHO BOOKS AGENT TESTS"
echo "======================================================================"

echo ""
echo "💰 Test 1: Invoice Query"
RESULT=$(curl -s -X POST "$BASE_URL/api/chat" \
    -H "Content-Type: application/json" \
    -d "{
        \"prompt\": \"Show me all unpaid invoices from Zoho Books\"
    }")

if echo "$RESULT" | grep -q '"final_response"\|"task_agent_pairs"'; then
    print_result "zoho_books" "Invoice Query" "true" ""
else
    print_result "zoho_books" "Invoice Query" "false" "$(echo $RESULT | head -c 100)"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "======================================================================"
echo "  TEST SUMMARY"
echo "======================================================================"
echo ""
echo "  Tests completed. Check results above for details."
echo ""
echo "  Note: Some tests may require API keys or credentials:"
echo "    - Mail Agent: Requires COMPOSIO_API_KEY"
echo "    - Zoho Books Agent: Requires Zoho OAuth tokens"
echo ""
echo "======================================================================"

# Cleanup
rm -f $TEST_CSV $TEST_DOC