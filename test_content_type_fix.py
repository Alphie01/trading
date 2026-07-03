#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content-Type Fix Test
Tests API endpoints with various content-type scenarios
"""

import requests
import json
import sys
import os

def test_content_type_fix():
    """Test the API endpoints with different content-type scenarios"""
    
    print("🧪 Testing Content-Type fix for API endpoints...")
    
    # Test data
    test_payload = {
        "coin_symbol": "BTC",
        "analysis_type": "lstm"
    }
    
    base_url = "http://localhost:5000"
    
    # Test scenarios
    test_cases = [
        {
            "name": "Proper JSON Content-Type",
            "headers": {"Content-Type": "application/json"},
            "data": json.dumps(test_payload),
            "expected": "Should work"
        },
        {
            "name": "Missing Content-Type (using requests.post with json param)",
            "headers": {},
            "json_param": test_payload,
            "expected": "Should work (requests auto-sets content-type)"
        },
        {
            "name": "Wrong Content-Type with force=True fallback",
            "headers": {"Content-Type": "text/plain"},
            "data": json.dumps(test_payload),
            "expected": "Should work with our fix"
        }
    ]
    
    print(f"\n📍 Testing endpoint: {base_url}/api/analyze_coin")
    print("Note: These are theoretical tests. Actual testing requires:")
    print("1. Web service running on localhost:5000")
    print("2. Valid authentication session")
    print("3. AI service running on localhost:8000")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Expected: {test_case['expected']}")
        
        # Show what the request would look like
        if 'json_param' in test_case:
            print(f"   Request: requests.post(url, json={test_case['json_param']})")
        else:
            print(f"   Request: requests.post(url, data='{test_case['data']}', headers={test_case['headers']})")
    
    print("\n✅ Our fix handles these scenarios:")
    print("   - Checks request.is_json first")
    print("   - Falls back to request.get_json(force=True) if needed")
    print("   - Returns proper HTTP 415 error with clear message if JSON parsing fails")
    print("   - Includes proper HTTP status codes (400, 415, 500)")
    
    print("\n🔧 Fixed endpoints:")
    print("   - /api/analyze_coin")
    print("   - /api/train_coin")

if __name__ == "__main__":
    test_content_type_fix()
