import time
import threading
import logging
import sys
import os

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

    def start(self):

        print("Fusion Engine Running\n")

        while not self.stop_event.is_set():

            P_key = self.key_logger.get_probability()
            P_face = self.face_logger.get_probability()

            P_total = (self.W_KEY * P_key) + (self.W_FACE * P_face)
            P_total = max(0.0, min(1.0, P_total))

            frustrated = P_total >= self.THRESHOLD

            print({
                "P_keystroke": round(P_key, 3),
                "P_face": round(P_face, 3),
                "P_total": round(P_total, 3),
                "frustrated": frustrated
            })

            if frustrated and time.time() - self.last_ai_time > self.ai_cooldown:

                print("\n" + "="*50)
                print("🔴 FRUSTRATION DETECTED! Launching Support System...")
                print("="*50 + "\n")

                # 1. Launch AI Assistant (Tkinter popup)
                print("🤖 Launching CodeBuddy AI Assistant...")
                suggestion = ask_ai(
                    "A developer is feeling frustrated while coding. Give a short, encouraging debugging tip to help them get unstuck."
                )
                
                # Run AI popup in separate thread
                threading.Thread(
                    target=show_cute_popup,
                    args=(suggestion,),
                    daemon=True
                ).start()
                
                # Small delay to ensure popup appears
                time.sleep(0.5)
                
                # 2. Launch Bug Smasher Game
                print("🎮 Launching Bug Smasher Game...")
                
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
                
                # Update cooldown timer
                self.last_ai_time = time.time()
                
                print("✅ Support system launched! (AI Assistant + Game)\n")

            time.sleep(1)