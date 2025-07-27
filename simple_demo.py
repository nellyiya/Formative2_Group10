#!/usr/bin/env python3
"""
Simplified Multimodal Authentication Demo
========================================

A streamlined version of the authentication system that focuses on 
demonstrating the core functionality without complex dependencies.
"""

import os
import pandas as pd
import numpy as np
import random
from pathlib import Path

class SimpleAuthDemo:
    """Simplified authentication system for demonstration"""
    
    def __init__(self):
        self.authorized_users = ['Branis', 'Tanguy', 'Nelly', 'Nhial']
        self.setup_sample_data()
        print("✅ Simple Authentication System Initialized")
    
    def setup_sample_data(self):
        """Create sample user profiles and purchase data"""
        # Sample user profiles with biometric "signatures"
        self.user_profiles = {
            'Branis': {
                'face_signature': np.random.RandomState(1).rand(10),
                'voice_signature': np.random.RandomState(11).rand(5),
                'purchase_history': {'category': 'Sports', 'avg_amount': 350, 'rating': 4.2}
            },
            'Tanguy': {
                'face_signature': np.random.RandomState(2).rand(10),
                'voice_signature': np.random.RandomState(12).rand(5),
                'purchase_history': {'category': 'Electronics', 'avg_amount': 450, 'rating': 3.8}
            },
            'Nelly': {
                'face_signature': np.random.RandomState(3).rand(10),
                'voice_signature': np.random.RandomState(13).rand(5),
                'purchase_history': {'category': 'Clothing', 'avg_amount': 280, 'rating': 4.5}
            },
            'Nhial': {
                'face_signature': np.random.RandomState(4).rand(10),
                'voice_signature': np.random.RandomState(14).rand(5),
                'purchase_history': {'category': 'Books', 'avg_amount': 180, 'rating': 3.9}
            }
        }
        
        # Product recommendations by category
        self.recommendations = {
            'Sports': ['Premium Running Shoes', 'Fitness Tracker', 'Protein Supplements'],
            'Electronics': ['Latest Smartphone', 'Wireless Headphones', 'Smart Watch'],
            'Clothing': ['Designer Jacket', 'Luxury Handbag', 'Fashion Accessories'],
            'Books': ['Bestseller Collection', 'E-Reader', 'Online Course Bundle']
        }
    
    def simulate_face_recognition(self, user_name, is_genuine=True):
        """Simulate face recognition process"""
        print(f"🔍 Processing face image for user: {user_name}")
        
        if is_genuine and user_name in self.authorized_users:
            # Simulate successful recognition with some noise
            actual_signature = self.user_profiles[user_name]['face_signature']
            simulated_input = actual_signature + np.random.normal(0, 0.1, len(actual_signature))
            
            # Calculate similarity
            similarity = 1 - np.mean(np.abs(actual_signature - simulated_input))
            confidence = max(0.7, min(0.95, similarity))
            
            print(f"   👤 Detected user: {user_name}")
            print(f"   📊 Confidence: {confidence:.1%}")
            
            return confidence > 0.6, user_name, confidence
        else:
            # Simulate failed recognition
            random_signature = np.random.rand(10)
            confidence = random.uniform(0.1, 0.4)
            
            print(f"   ❓ Unrecognized face pattern")
            print(f"   📊 Confidence: {confidence:.1%}")
            
            return False, "Unknown", confidence
    
    def simulate_voice_verification(self, user_name, is_genuine=True):
        """Simulate voice verification process"""
        print(f"🎤 Processing voice sample for user: {user_name}")
        
        if is_genuine and user_name in self.authorized_users:
            # Simulate successful verification
            actual_signature = self.user_profiles[user_name]['voice_signature']
            simulated_input = actual_signature + np.random.normal(0, 0.05, len(actual_signature))
            
            # Calculate similarity
            similarity = 1 - np.mean(np.abs(actual_signature - simulated_input))
            confidence = max(0.75, min(0.98, similarity))
            
            print(f"   🗣️  Voice pattern matches: {user_name}")
            print(f"   📊 Confidence: {confidence:.1%}")
            
            return confidence > 0.7, user_name, confidence
        else:
            # Simulate failed verification
            confidence = random.uniform(0.1, 0.5)
            
            print(f"   ❓ Voice pattern not recognized")
            print(f"   📊 Confidence: {confidence:.1%}")
            
            return False, "Unknown", confidence
    
    def generate_recommendations(self, user_name):
        """Generate personalized product recommendations"""
        if user_name not in self.authorized_users:
            return []
        
        profile = self.user_profiles[user_name]
        category = profile['purchase_history']['category']
        avg_amount = profile['purchase_history']['avg_amount']
        
        print(f"🛒 Generating recommendations for {user_name}")
        print(f"   💰 Average purchase: ${avg_amount}")
        print(f"   🎯 Preferred category: {category}")
        print(f"   🔮 Recommended products:")
        
        products = self.recommendations.get(category, ['General Products'])
        for i, product in enumerate(products, 1):
            price = avg_amount + random.randint(-50, 100)
            print(f"      {i}. {product} - ${price}")
        
        return products
    
    def run_transaction(self, scenario_name, user_name, is_authorized=True):
        """Run a complete transaction simulation"""
        print(f"\n{'='*60}")
        print(f"🔐 TRANSACTION: {scenario_name}")
        print(f"{'='*60}")
        print(f"👤 User: {user_name}")
        print(f"🎭 Expected: {'✅ AUTHORIZED' if is_authorized else '❌ UNAUTHORIZED'}")
        print(f"{'='*60}")
        
        # Step 1: Face Recognition
        print(f"\n🔸 STEP 1: Face Recognition")
        print("-" * 30)
        
        face_success, face_user, face_conf = self.simulate_face_recognition(user_name, is_authorized)
        
        if not face_success:
            print("❌ FACE AUTHENTICATION FAILED")
            print("🚫 ACCESS DENIED - Transaction terminated")
            print(f"💡 Reason: Face not recognized or confidence too low")
            return False
        
        print("✅ FACE AUTHENTICATION SUCCESSFUL")
        print("🟢 Proceeding to voice verification...")
        
        # Step 2: Voice Verification
        print(f"\n🔸 STEP 2: Voice Verification")
        print("-" * 30)
        
        voice_success, voice_user, voice_conf = self.simulate_voice_verification(user_name, is_authorized)
        
        if not voice_success:
            print("❌ VOICE VERIFICATION FAILED")
            print("🚫 ACCESS DENIED - Transaction terminated")
            print(f"💡 Reason: Voice pattern not recognized")
            return False
        
        print("✅ VOICE VERIFICATION SUCCESSFUL")
        print("🟢 Multimodal authentication complete!")
        
        # Step 3: Product Recommendations
        print(f"\n🔸 STEP 3: Product Recommendations")
        print("-" * 30)
        
        recommendations = self.generate_recommendations(user_name)
        
        print(f"\n✅ TRANSACTION COMPLETED SUCCESSFULLY")
        print(f"🎉 Welcome {user_name}! Access granted to recommendation system.")
        
        return True
    
    def run_demo(self):
        """Run the complete demonstration"""
        print("🎯 MULTIMODAL AUTHENTICATION SYSTEM DEMO")
        print("🔒 Simulating face recognition + voice verification")
        print("🛍️  Product recommendation system\n")
        
        # Define test scenarios
        scenarios = [
            {
                'name': 'Authorized User - Branis',
                'user': 'Branis',
                'authorized': True,
                'description': 'Legitimate user with valid biometrics'
            },
            {
                'name': 'Authorized User - Nelly',
                'user': 'Nelly',
                'authorized': True,
                'description': 'Another legitimate user'
            },
            {
                'name': 'Unauthorized Attempt - Face Spoofing',
                'user': 'Branis',
                'authorized': False,
                'description': 'Attempted face spoofing attack'
            },
            {
                'name': 'Unauthorized Attempt - Unknown User',
                'user': 'Unknown_Person',
                'authorized': False,
                'description': 'Completely unknown individual'
            }
        ]
        
        # Run each scenario
        for i, scenario in enumerate(scenarios, 1):
            result = self.run_transaction(
                scenario['name'],
                scenario['user'],
                scenario['authorized']
            )
            
            print(f"\n🏁 SCENARIO {i} RESULT: {'✅ SUCCESS' if result else '❌ BLOCKED'}")
            print(f"💬 {scenario['description']}")
            
            if i < len(scenarios):
                input(f"\n⏳ Press Enter to continue to scenario {i+1}...")
        
        # Summary
        print(f"\n{'='*60}")
        print("🎊 DEMONSTRATION COMPLETED")
        print("💡 System Features Demonstrated:")
        print("   🔹 Multimodal biometric authentication")
        print("   🔹 Security against unauthorized access")
        print("   🔹 Personalized product recommendations")
        print("   🔹 Real-time transaction processing")
        print("="*60)

def main():
    """Main function"""
    try:
        print("🚀 Initializing Simple Authentication Demo...")
        demo = SimpleAuthDemo()
        demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n👋 Thank you for the demonstration!")

if __name__ == "__main__":
    main()
