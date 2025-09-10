# Green Track AI System - Flask API Backend
# Carbon Footprint Monitoring with DeepSeek AI

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import sqlite3
import re
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSeekService:
    """Service for interacting with DeepSeek API via OpenRouter with rate limiting"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "deepseek/deepseek-r1-distill-llama-70b:free"
        self.last_request_time = 0
        self.min_request_interval = 2  # Minimum 2 seconds between requests
        self.request_lock = threading.Lock()
        
    def chat_completion(self, messages: List[Dict], temperature: float = 0.7, max_retries: int = 3) -> Dict:
        """Send chat completion request to DeepSeek via OpenRouter with rate limiting and retries"""
        
        # Rate limiting
        with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
            self.last_request_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/green-track",
            "X-Title": "Green Track AI"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 429:
                    # Rate limited, wait longer
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                    logger.warning(f"Rate limited, waiting {wait_time} seconds before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.error(f"DeepSeek API error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return {"error": f"API request failed after {max_retries} attempts: {str(e)}"}
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {"error": "Max retries exceeded"}

class DatabaseManager:
    """Handle SQLite database operations with proper connection management"""
    
    def __init__(self, db_path: str = "green_track.db"):
        self.db_path = db_path
        self.db_lock = threading.Lock()
        self.init_database()
        
    def get_connection(self):
        """Get database connection with proper settings"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")  # Better for concurrent access
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=memory")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        return conn
        
    def init_database(self):
        """Initialize database with required tables"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Activities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    category TEXT NOT NULL,
                    subcategory TEXT,
                    description TEXT,
                    quantity REAL,
                    unit TEXT,
                    co2_estimate REAL,
                    confidence TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Recommendations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    recommendation TEXT,
                    impact_score REAL,
                    difficulty INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
            conn.close()
        
    def add_user(self, username: str, email: str) -> Optional[int]:
        """Add new user and return user_id"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute(
                    "INSERT INTO users (username, email) VALUES (?, ?)",
                    (username, email)
                )
                user_id = cursor.lastrowid
                conn.commit()
                return user_id
            except sqlite3.IntegrityError:
                # User already exists, get their ID
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                result = cursor.fetchone()
                return result[0] if result else None
            except Exception as e:
                logger.error(f"Database error in add_user: {e}")
                return None
            finally:
                conn.close()
                
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute("SELECT id, username, email, created_at FROM users WHERE username = ?", (username,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        "id": result[0],
                        "username": result[1],
                        "email": result[2],
                        "created_at": result[3]
                    }
                return None
            except Exception as e:
                logger.error(f"Database error in get_user_by_username: {e}")
                return None
            finally:
                conn.close()
                
    def add_activity(self, user_id: int, activity_data: Dict) -> Optional[int]:
        """Add activity to database"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO activities 
                    (user_id, category, subcategory, description, quantity, unit, co2_estimate, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    activity_data.get('category'),
                    activity_data.get('subcategory'),
                    activity_data.get('description'),
                    activity_data.get('quantity'),
                    activity_data.get('unit'),
                    activity_data.get('co2_estimate'),
                    activity_data.get('confidence')
                ))
                
                activity_id = cursor.lastrowid
                conn.commit()
                return activity_id
            except Exception as e:
                logger.error(f"Database error in add_activity: {e}")
                return None
            finally:
                conn.close()
        
    def get_user_activities(self, user_id: int, days: int = 30) -> List[Dict]:
        """Get user activities from last N days"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                query = '''
                    SELECT id, category, subcategory, description, quantity, unit, co2_estimate, confidence, created_at
                    FROM activities 
                    WHERE user_id = ? AND created_at >= datetime('now', '-{} days')
                    ORDER BY created_at DESC
                '''.format(days)
                
                cursor.execute(query, (user_id,))
                results = cursor.fetchall()
                
                activities = []
                for row in results:
                    activities.append({
                        "id": row[0],
                        "category": row[1],
                        "subcategory": row[2],
                        "description": row[3],
                        "quantity": row[4],
                        "unit": row[5],
                        "co2_estimate": row[6],
                        "confidence": row[7],
                        "created_at": row[8]
                    })
                
                return activities
            except Exception as e:
                logger.error(f"Database error in get_user_activities: {e}")
                return []
            finally:
                conn.close()

class CarbonFootprintAnalyzer:
    """Main AI engine for carbon footprint analysis"""
    
    def __init__(self, deepseek_service: DeepSeekService):
        self.deepseek = deepseek_service
        self.emission_factors = {
            'transport': {
                'car_petrol': 0.21,  # kg CO2 per km
                'car_diesel': 0.18,
                'bus': 0.08,
                'train': 0.04,
                'flight_domestic': 0.25,
                'flight_international': 0.3
            },
            'energy': {
                'electricity': 0.5,  # kg CO2 per kWh
                'natural_gas': 2.0,  # kg CO2 per cubic meter
                'heating_oil': 2.5   # kg CO2 per liter
            },
            'food': {
                'beef': 60,  # kg CO2 per kg
                'chicken': 6,
                'fish': 5,
                'vegetables': 2,
                'dairy': 3
            }
        }
        
    def parse_activity(self, user_input: str) -> Dict:
        """Parse natural language activity description"""
        prompt = f"""
        You are a carbon footprint expert. Parse this activity description:
        "{user_input}"
        
        Extract and return ONLY a valid JSON object with this exact structure:
        {{
          "category": "transport|energy|food|consumption",
          "subcategory": "specific type (e.g., car_petrol, electricity, beef)",
          "quantity": numeric_value_only,
          "unit": "km|kWh|kg|liters|etc",
          "description": "clean description of the activity",
          "confidence": "high|medium|low"
        }}
        
        Examples:
        - "drove 50 km in my car" -> {{"category": "transport", "subcategory": "car_petrol", "quantity": 50, "unit": "km", "description": "car travel", "confidence": "high"}}
        - "used 15 kWh electricity" -> {{"category": "energy", "subcategory": "electricity", "quantity": 15, "unit": "kWh", "description": "electricity consumption", "confidence": "high"}}
        
        Return only the JSON object, no other text.
        """
        
        response = self.deepseek.chat_completion([
            {"role": "user", "content": prompt}
        ])
        
        # Check if API failed due to rate limiting
        if "error" in response:
            return {"error": f"API temporarily unavailable: {response['error']}"}
        
        try:
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                return parsed_data
            else:
                return {"error": "Could not parse activity"}
        except Exception as e:
            logger.error(f"Activity parsing error: {e}")
            return {"error": f"Parsing failed: {str(e)}"}
            
    def estimate_emissions(self, activity_data: Dict) -> float:
        """Estimate CO2 emissions for an activity"""
        category = activity_data.get('category')
        subcategory = activity_data.get('subcategory')
        quantity = activity_data.get('quantity', 0)
        
        if category in self.emission_factors and subcategory in self.emission_factors[category]:
            factor = self.emission_factors[category][subcategory]
            return quantity * factor
        
        # Use DeepSeek for unknown activities
        prompt = f"""
        Estimate CO2 emissions for this activity:
        Category: {category}
        Subcategory: {subcategory}
        Quantity: {quantity} {activity_data.get('unit', '')}
        
        Return only a number representing kg CO2 equivalent.
        Base your estimate on standard emission factors.
        """
        
        response = self.deepseek.chat_completion([
            {"role": "user", "content": prompt}
        ])
        
        # Check if API failed
        if "error" in response:
            # Use simple fallback calculation
            fallback_factors = {
                'transport': 0.2,  # kg CO2 per km
                'energy': 0.5,     # kg CO2 per kWh  
                'food': 10,        # kg CO2 per kg
                'consumption': 2   # kg CO2 per item
            }
            factor = fallback_factors.get(category, 1.0)
            return max(0.1, quantity * factor)
        
        try:
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '0')
            # Extract number from response
            number_match = re.search(r'(\d+\.?\d*)', content)
            if number_match:
                return float(number_match.group(1))
            return 1.0  # Default fallback
        except:
            return 1.0
            
    def generate_recommendations(self, user_activities: List[Dict]) -> List[Dict]:
        """Generate personalized recommendations based on user data"""
        if not user_activities:
            return [{"recommendation": "Start logging activities to get personalized recommendations!", "impact_score": 0, "difficulty": 1}]
        
        # Analyze user patterns
        total_emissions = sum(activity.get('co2_estimate', 0) for activity in user_activities)
        
        # Group by category
        category_emissions = {}
        for activity in user_activities:
            category = activity.get('category', 'unknown')
            category_emissions[category] = category_emissions.get(category, 0) + activity.get('co2_estimate', 0)
        
        # Create activity summary for prompt
        activity_summary = "\n".join([
            f"- {activity.get('category', 'N/A')}: {activity.get('subcategory', 'N/A')} "
            f"({activity.get('quantity', 0)} {activity.get('unit', '')}) = {activity.get('co2_estimate', 0):.2f} kg CO2"
            for activity in user_activities[:10]
        ])
        
        prompt = f"""
        Analyze this user's carbon footprint data and provide 3 specific recommendations:
        
        Total emissions (last 30 days): {total_emissions:.2f} kg CO2
        Emissions by category: {category_emissions}
        
        Recent activities:
        {activity_summary}
        
        Provide 3 actionable recommendations in this exact JSON format:
        [
          {{
            "recommendation": "specific action to take",
            "impact_score": number_from_1_to_10,
            "difficulty": number_from_1_to_5
          }}
        ]
        
        Focus on:
        1. Highest emission categories
        2. Easy wins for beginners
        3. Long-term behavior changes
        
        Return only the JSON array, no other text.
        """
        
        response = self.deepseek.chat_completion([
            {"role": "user", "content": prompt}
        ])
        
        try:
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '[]')
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group())
                return recommendations
            return []
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return [{"recommendation": "Error generating recommendations", "impact_score": 0, "difficulty": 1}]

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize services
API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
if not API_KEY:
    logger.error("DEEPSEEK_API_KEY environment variable not set!")

deepseek_service = DeepSeekService(API_KEY)
db_manager = DatabaseManager()
analyzer = CarbonFootprintAnalyzer(deepseek_service)

# API Routes

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Green Track AI API",
        "version": "1.0.0"
    })

@app.route('/api/register', methods=['POST'])
def register_user():
    """Register or login user"""
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        
        if not username or not email:
            return jsonify({"error": "Username and email are required"}), 400
        
        user_id = db_manager.add_user(username, email)
        if user_id:
            user = db_manager.get_user_by_username(username)
            return jsonify({
                "success": True,
                "message": f"Welcome {username}! You are now registered.",
                "user": user
            })
        
        return jsonify({"error": "Error registering user"}), 500
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/<username>', methods=['GET'])
def get_user(username):
    """Get user by username"""
    try:
        user = db_manager.get_user_by_username(username)
        if user:
            return jsonify({"success": True, "user": user})
        return jsonify({"error": "User not found"}), 404
    except Exception as e:
        logger.error(f"Get user error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/activity', methods=['POST'])
def log_activity():
    """Log a new activity"""
    try:
        data = request.get_json()
        username = data.get('username')
        activity_description = data.get('activity_description')
        
        if not username or not activity_description:
            return jsonify({"error": "Username and activity description are required"}), 400
        
        # Get user
        user = db_manager.get_user_by_username(username)
        if not user:
            return jsonify({"error": "User not found. Please register first."}), 404
        
        # Parse activity using AI
        parsed_data = analyzer.parse_activity(activity_description)
        
        if "error" in parsed_data:
            return jsonify({"error": f"Error parsing activity: {parsed_data['error']}"}), 400
        
        # Estimate emissions
        co2_estimate = analyzer.estimate_emissions(parsed_data)
        parsed_data['co2_estimate'] = co2_estimate
        
        # Save to database
        activity_id = db_manager.add_activity(user['id'], parsed_data)
        
        if not activity_id:
            return jsonify({"error": "Failed to save activity to database"}), 500
        
        return jsonify({
            "success": True,
            "message": "Activity logged successfully!",
            "analysis": {
                "activity_id": activity_id,
                "category": parsed_data.get('category'),
                "subcategory": parsed_data.get('subcategory'),
                "quantity": parsed_data.get('quantity'),
                "unit": parsed_data.get('unit'),
                "co2_estimate": co2_estimate,
                "confidence": parsed_data.get('confidence')
            }
        })
        
    except Exception as e:
        logger.error(f"Activity logging error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/activities/<username>', methods=['GET'])
def get_user_activities(username):
    """Get user activities"""
    try:
        days = request.args.get('days', 30, type=int)
        
        # Get user
        user = db_manager.get_user_by_username(username)
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        activities = db_manager.get_user_activities(user['id'], days)
        
        return jsonify({
            "success": True,
            "activities": activities,
            "total_activities": len(activities)
        })
        
    except Exception as e:
        logger.error(f"Get activities error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard/<username>', methods=['GET'])
def get_dashboard_data(username):
    """Get dashboard data for user"""
    try:
        days = request.args.get('days', 30, type=int)
        
        # Get user
        user = db_manager.get_user_by_username(username)
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        activities = db_manager.get_user_activities(user['id'], days)
        
        if not activities:
            return jsonify({
                "success": True,
                "message": "No activities logged yet",
                "dashboard_data": {
                    "total_emissions": 0,
                    "avg_daily_emissions": 0,
                    "total_activities": 0,
                    "category_breakdown": {},
                    "daily_trends": []
                }
            })
        
        # Calculate summary statistics
        total_emissions = sum(activity.get('co2_estimate', 0) for activity in activities)
        avg_daily = total_emissions / days
        
        # Category breakdown
        category_breakdown = {}
        for activity in activities:
            category = activity.get('category', 'unknown')
            category_breakdown[category] = category_breakdown.get(category, 0) + activity.get('co2_estimate', 0)
        
        # Daily trends (simplified)
        daily_trends = {}
        for activity in activities:
            date = activity.get('created_at', '').split(' ')[0]  # Extract date part
            daily_trends[date] = daily_trends.get(date, 0) + activity.get('co2_estimate', 0)
        
        # Convert to list for frontend
        daily_trends_list = [{"date": date, "emissions": emissions} for date, emissions in daily_trends.items()]
        daily_trends_list.sort(key=lambda x: x['date'])
        
        return jsonify({
            "success": True,
            "dashboard_data": {
                "total_emissions": round(total_emissions, 2),
                "avg_daily_emissions": round(avg_daily, 2),
                "total_activities": len(activities),
                "category_breakdown": {k: round(v, 2) for k, v in category_breakdown.items()},
                "daily_trends": daily_trends_list
            }
        })
        
    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommendations/<username>', methods=['GET'])
def get_recommendations(username):
    """Get personalized recommendations"""
    try:
        # Get user
        user = db_manager.get_user_by_username(username)
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        activities = db_manager.get_user_activities(user['id'], days=30)
        recommendations = analyzer.generate_recommendations(activities)
        
        return jsonify({
            "success": True,
            "recommendations": recommendations
        })
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_activity():
    """Analyze activity without saving (for testing)"""
    try:
        data = request.get_json()
        activity_description = data.get('activity_description')
        
        if not activity_description:
            return jsonify({"error": "Activity description is required"}), 400
        
        # Parse activity using AI
        parsed_data = analyzer.parse_activity(activity_description)
        
        if "error" in parsed_data:
            return jsonify({"error": f"Error parsing activity: {parsed_data['error']}"}), 400
        
        # Estimate emissions
        co2_estimate = analyzer.estimate_emissions(parsed_data)
        parsed_data['co2_estimate'] = co2_estimate
        
        return jsonify({
            "success": True,
            "analysis": parsed_data
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
