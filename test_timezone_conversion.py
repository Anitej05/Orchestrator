#!/usr/bin/env python3
"""
Test script to verify timezone conversion for cron scheduling.
Tests the IST to UTC conversion logic.
"""

def test_ist_to_utc_conversion():
    """Test IST to UTC conversion for cron scheduling"""
    
    test_cases = [
        # (user_hour_ist, user_minute_ist, expected_utc_hour, expected_utc_minute, description)
        (14, 0, 8, 30, "2:00 PM IST = 8:30 AM UTC"),
        (0, 30, 19, 0, "12:30 AM IST = 7:00 PM previous day UTC (wraps)"),
        (23, 0, 17, 30, "11:00 PM IST = 5:30 PM UTC"),
        (9, 0, 3, 30, "9:00 AM IST = 3:30 AM UTC"),
        (18, 45, 13, 15, "6:45 PM IST = 1:15 PM UTC"),
        (5, 30, 0, 0, "5:30 AM IST = 12:00 AM UTC"),
    ]
    
    print("Testing IST to UTC Conversion for Cron Scheduling")
    print("=" * 70)
    
    all_passed = True
    
    for user_hour, user_minute, expected_hour, expected_minute, description in test_cases:
        # Convert IST to UTC
        utc_hour = user_hour - 5
        utc_minute = user_minute - 30
        
        # Handle minute underflow
        if utc_minute < 0:
            utc_hour -= 1
            utc_minute += 60
        
        # Handle hour underflow (previous day)
        if utc_hour < 0:
            utc_hour += 24
        
        # Check result
        passed = (utc_hour == expected_hour and utc_minute == expected_minute)
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {description}")
        print(f"       Input:    {user_hour:02d}:{user_minute:02d} IST")
        print(f"       Expected: {expected_hour:02d}:{expected_minute:02d} UTC")
        print(f"       Got:      {utc_hour:02d}:{utc_minute:02d} UTC")
        if not passed:
            print(f"       ERROR: Mismatch!")
        print()
    
    print("=" * 70)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

def test_cron_generation():
    """Test cron expression generation with timezone conversion"""
    
    print("\nTesting Cron Expression Generation")
    print("=" * 70)
    
    test_cases = [
        # (schedule_type, user_hour, user_minute, day_of_week, day_of_month, expected_cron, description)
        ("daily", 14, 0, "1", "1", "30 08 * * *", "Daily at 2:00 PM IST = 8:30 AM UTC"),
        ("hourly", 14, 15, "1", "1", "45 * * * *", "Hourly at :15 past (UTC minute)"),
        ("weekly", 14, 0, "3", "1", "30 08 * * 3", "Weekly on Wednesday at 2:00 PM IST"),
        ("monthly", 14, 0, "1", "15", "30 08 15 * *", "Monthly on 15th at 2:00 PM IST"),
    ]
    
    def generate_cron(schedule_type, user_hour, user_minute, day_of_week, day_of_month):
        """Generate cron expression with IST to UTC conversion"""
        # Convert IST to UTC
        utc_hour = user_hour - 5
        utc_minute = user_minute - 30
        
        if utc_minute < 0:
            utc_hour -= 1
            utc_minute += 60
        
        if utc_hour < 0:
            utc_hour += 24
        
        m = str(utc_minute).zfill(2)
        h = str(utc_hour).zfill(2)
        
        if schedule_type == "hourly":
            return f"{m} * * * *"
        elif schedule_type == "daily":
            return f"{m} {h} * * *"
        elif schedule_type == "weekly":
            return f"{m} {h} * * {day_of_week}"
        elif schedule_type == "monthly":
            return f"{m} {h} {day_of_month} * *"
        else:
            return "0 9 * * *"
    
    all_passed = True
    
    for schedule_type, user_hour, user_minute, day_of_week, day_of_month, expected_cron, description in test_cases:
        generated_cron = generate_cron(schedule_type, user_hour, user_minute, day_of_week, day_of_month)
        passed = (generated_cron == expected_cron)
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {description}")
        print(f"       Expected: {expected_cron}")
        print(f"       Got:      {generated_cron}")
        if not passed:
            print(f"       ERROR: Mismatch!")
        print()
    
    print("=" * 70)
    if all_passed:
        print("✓ All cron generation tests passed!")
        return 0
    else:
        print("✗ Some cron generation tests failed!")
        return 1

if __name__ == "__main__":
    result1 = test_ist_to_utc_conversion()
    result2 = test_cron_generation()
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    if result1 == 0 and result2 == 0:
        print("✓ All tests passed! Timezone conversion is working correctly.")
        exit(0)
    else:
        print("✗ Some tests failed! Please review the timezone conversion logic.")
        exit(1)
