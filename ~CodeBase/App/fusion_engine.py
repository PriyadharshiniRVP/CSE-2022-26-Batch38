import time
import threading
import logging
import sys
import os
import pandas as pd
from datetime import datetime
from collections import deque

from nova_assistant import ask_ai
from cute_popup import show_cute_popup

# Add parent directory to path for Game module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Game.bug_smasher import launch_game

# suppress LightGBM warnings
logging.getLogger("lightgbm").setLevel(logging.CRITICAL)


class FusionEngine:
    def __init__(self, keystroke_logger, facial_logger, stop_event):
        self.W_KEY = 0.65
        self.W_FACE = 0.35
        self.THRESHOLD = 0.6

        self.key_logger = keystroke_logger
        self.face_logger = facial_logger
        self.stop_event = stop_event

        # cooldown timer for AI assistant
        self.last_ai_time = 0
        self.ai_cooldown = 60  # 60 seconds cooldown
        
        # Track session points
        self.session_points = 0
        
        # ========== METRICS TRACKING ==========
        # Store predictions for metrics calculation
        self.predictions = []  # Store (timestamp, P_total, frustrated)
        self.interventions = []  # Store intervention times
        self.frustration_episodes = []  # Store frustration periods
        self.current_episode_start = None
        
        # Session metadata
        self.session_start_time = time.time()
        self.total_time = 0
        self.total_frustration_time = 0
        self.intervention_count = 0
        
        # For real-time display
        self.metrics_history = deque(maxlen=100)  # Last 100 predictions
        
        # Create metrics file header
        self.metrics_file = f"metrics_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_metrics_file()

    def init_metrics_file(self):
        """Initialize metrics CSV file with headers"""
        with open(self.metrics_file, 'w') as f:
            f.write("timestamp,P_keystroke,P_face,P_total,is_frustrated,intervention_triggered\n")
        print(f"📊 Metrics will be saved to: {self.metrics_file}")

    def log_metrics(self, P_key, P_face, P_total, frustrated, intervention_triggered=False):
        """Log metrics to CSV file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.metrics_file, 'a') as f:
            f.write(f"{timestamp},{P_key:.4f},{P_face:.4f},{P_total:.4f},{int(frustrated)},{int(intervention_triggered)}\n")
        
        # Store for in-memory calculations
        self.predictions.append({
            'timestamp': time.time(),
            'P_total': P_total,
            'frustrated': frustrated
        })
        
        # Keep only last 1000 predictions to save memory
        if len(self.predictions) > 1000:
            self.predictions = self.predictions[-1000:]
        
        # Track frustration episodes
        if frustrated and self.current_episode_start is None:
            self.current_episode_start = time.time()
        elif not frustrated and self.current_episode_start is not None:
            episode_duration = time.time() - self.current_episode_start
            self.frustration_episodes.append(episode_duration)
            self.total_frustration_time += episode_duration
            self.current_episode_start = None

    def calculate_metrics(self):
        """Calculate comprehensive metrics from collected data"""
        if not self.predictions:
            return {}
        
        total_predictions = len(self.predictions)
        frustrated_count = sum(1 for p in self.predictions if p['frustrated'])
        
        # Basic metrics
        frustration_rate = frustrated_count / total_predictions if total_predictions > 0 else 0
        
        # Calculate total session time
        self.total_time = time.time() - self.session_start_time
        
        # Frustration percentage
        frustration_percentage = (self.total_frustration_time / self.total_time * 100) if self.total_time > 0 else 0
        
        # Average frustration episode duration
        avg_episode_duration = sum(self.frustration_episodes) / len(self.frustration_episodes) if self.frustration_episodes else 0
        
        # Intervention frequency (interventions per hour)
        intervention_frequency = (self.intervention_count / self.total_time * 3600) if self.total_time > 0 else 0
        
        # Calculate P_total statistics
        p_total_values = [p['P_total'] for p in self.predictions]
        
        metrics = {
            'session_duration_minutes': round(self.total_time / 60, 2),
            'total_predictions': total_predictions,
            'frustrated_count': frustrated_count,
            'frustration_rate': round(frustration_rate * 100, 2),
            'total_frustration_time_seconds': round(self.total_frustration_time, 2),
            'frustration_percentage': round(frustration_percentage, 2),
            'frustration_episodes': len(self.frustration_episodes),
            'avg_episode_duration_seconds': round(avg_episode_duration, 2),
            'interventions_triggered': self.intervention_count,
            'interventions_per_hour': round(intervention_frequency, 2),
            'avg_P_total': round(sum(p_total_values) / len(p_total_values), 4),
            'max_P_total': round(max(p_total_values), 4),
            'min_P_total': round(min(p_total_values), 4),
            'session_points_earned': self.session_points
        }
        
        return metrics

    def print_metrics_summary(self):
        """Print a formatted metrics summary to console"""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*60)
        print("📊 SESSION METRICS SUMMARY")
        print("="*60)
        print(f"⏱️  Session Duration: {metrics['session_duration_minutes']} minutes")
        print(f"📈 Total Predictions: {metrics['total_predictions']}")
        print(f"😤 Frustration Rate: {metrics['frustration_rate']}%")
        print(f"⏰ Total Frustration Time: {metrics['total_frustration_time_seconds']} seconds")
        print(f"📊 Frustration Percentage: {metrics['frustration_percentage']}%")
        print(f"🔄 Frustration Episodes: {metrics['frustration_episodes']}")
        print(f"⏱️  Avg Episode Duration: {metrics['avg_episode_duration_seconds']} seconds")
        print(f"🆘 Interventions Triggered: {metrics['interventions_triggered']}")
        print(f"📞 Interventions/Hour: {metrics['interventions_per_hour']}")
        print(f"🎯 Avg Frustration Score: {metrics['avg_P_total']}")
        print(f"📈 Max Frustration Score: {metrics['max_P_total']}")
        print(f"📉 Min Frustration Score: {metrics['min_P_total']}")
        print(f"✨ Session Points Earned: {metrics['session_points_earned']}")
        print("="*60)
        
        # Save summary to file
        self.save_metrics_summary(metrics)

    def save_metrics_summary(self, metrics):
        """Save metrics summary to a separate file"""
        summary_file = f"metrics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("AI FRUSTRATION DETECTION SYSTEM - SESSION SUMMARY\n")
            f.write(f"Session Start: {datetime.fromtimestamp(self.session_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n💾 Metrics summary saved to: {summary_file}")

    def get_realtime_stats(self):
        """Get real-time statistics for display"""
        if len(self.metrics_history) == 0:
            return "No data yet"
        
        recent = list(self.metrics_history)[-10:]  # Last 10 predictions
        recent_frustrated = sum(1 for p in recent if p)
        
        return {
            'recent_frustration_rate': (recent_frustrated / len(recent) * 100) if recent else 0,
            'active_frustration': self.current_episode_start is not None,
            'current_streak': self._get_current_frustration_streak()
        }

    def _get_current_frustration_streak(self):
        """Get current consecutive frustration predictions"""
        streak = 0
        for p in reversed(self.predictions):
            if p['frustrated']:
                streak += 1
            else:
                break
        return streak

    def start(self):
        print("Fusion Engine Running\n")
        print("="*60)
        print("📊 Metrics Collection Active")
        print(f"📁 Metrics will be saved to: {self.metrics_file}")
        print("="*60 + "\n")

        while not self.stop_event.is_set():
            P_key = self.key_logger.get_probability()
            P_face = self.face_logger.get_probability()

            P_total = (self.W_KEY * P_key) + (self.W_FACE * P_face)
            P_total = max(0.0, min(1.0, P_total))

            frustrated = P_total >= self.THRESHOLD
            
            # Store for real-time display
            self.metrics_history.append(frustrated)
            
            # Check if intervention should be triggered
            intervention_triggered = False
            if frustrated and time.time() - self.last_ai_time > self.ai_cooldown:
                intervention_triggered = True
                self.intervention_count += 1
                self.last_ai_time = time.time()
                
                # Launch support system in background
                threading.Thread(target=self._launch_support_system, daemon=True).start()

            # Log metrics
            self.log_metrics(P_key, P_face, P_total, frustrated, intervention_triggered)

            # Enhanced console output with metrics
            realtime = self.get_realtime_stats()
            
            print({
                "P_keystroke": round(P_key, 3),
                "P_face": round(P_face, 3),
                "P_total": round(P_total, 3),
                "frustrated": frustrated,
                "streak": realtime['current_streak'],
                "recent_frustration": f"{realtime['recent_frustration_rate']:.0f}%"
            })

            if frustrated and intervention_triggered:
                print("\n" + "="*50)
                print("🔴 FRUSTRATION DETECTED! Launching Support System...")
                print(f"   Current Streak: {realtime['current_streak']}")
                print(f"   Recent Frustration: {realtime['recent_frustration_rate']:.0f}%")
                print("="*50 + "\n")

            time.sleep(1)

    def _launch_support_system(self):
        """Launch AI Assistant and Game in background"""
        # 1. Launch AI Assistant
        suggestion = ask_ai(
            "A developer is feeling frustrated while coding. Give a short, encouraging debugging tip to help them get unstuck."
        )
        
        # Run AI popup in separate thread
        threading.Thread(
            target=show_cute_popup,
            args=(suggestion,),
            daemon=True
        ).start()
        
        time.sleep(0.5)
        
        # 2. Launch Bug Smasher Game
        def on_game_end(results):
            """Callback when game finishes"""
            print(f"\n📊 Game Results:")
            print(f"   - Bugs smashed: {results['bugs_smashed']}")
            print(f"   - Focus energy: {results['focus_energy']}%")
            print(f"   - Score: {results['score']}")
            
            # Add focus energy to session points
            self.session_points += results['focus_energy']
            print(f"✨ Added {results['focus_energy']} focus points to session!")
            print(f"💰 Total session points: {self.session_points}\n")
        
        # Launch game in separate thread
        game_thread = threading.Thread(
            target=launch_game,
            args=(on_game_end,),
            daemon=True
        )
        game_thread.start()
        print("✅ Support system launched! (AI Assistant + Game)")

    def get_session_summary(self):
        """Public method to get session summary"""
        return self.calculate_metrics()