import pygame
import random
import math
import sys

# Initialize pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

# Colors
BLACK = (0, 0, 0)
DARK_PURPLE = (20, 10, 40)
NEON_PURPLE = (191, 79, 255)
NEON_BLUE = (0, 243, 255)
NEON_PINK = (255, 0, 110)
NEON_GREEN = (0, 255, 191)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
YELLOW = (255, 200, 80)
ORANGE = (255, 159, 28)

class Particle:
    """Simple particle effect"""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-3, 3)
        self.color = color
        self.life = 20
        self.size = random.randint(2, 4)
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life > 0
    
    def draw(self, screen):
        if self.life > 0:
            alpha = min(255, self.life * 12)
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)

class Bug:
    """Bug class"""
    def __init__(self, x, y, bug_type):
        self.x = x
        self.y = y
        self.type = bug_type
        
        # Bug properties
        if bug_type == 'syntax':
            self.color = RED
            self.points = 10
            self.hits_needed = 1
            self.speed = 3
            self.size = 30
            self.tip = "💡 Tip: Check your semicolons!"
        elif bug_type == 'logic':
            self.color = YELLOW
            self.points = 20
            self.hits_needed = 1
            self.speed = 2
            self.size = 35
            self.tip = "💡 Tip: Use print statements to debug!"
        elif bug_type == 'runtime':
            self.color = NEON_PURPLE
            self.points = 30
            self.hits_needed = 2
            self.speed = 1.5
            self.size = 40
            self.tip = "💡 Tip: Use try-catch blocks!"
        else:
            self.color = NEON_GREEN
            self.points = 50
            self.hits_needed = 2
            self.speed = 1.2
            self.size = 45
            self.tip = "💡 Tip: Close database connections!"
        
        self.hits_left = self.hits_needed
        self.speed_x = random.choice([-1, 1]) * self.speed
        self.speed_y = random.choice([-1, 1]) * self.speed
        self.angle = 0
    
    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y
        self.angle += 5
        
        # Bounce off walls
        if self.x < 50 or self.x > SCREEN_WIDTH - 50:
            self.speed_x *= -1
        if self.y < 80 or self.y > SCREEN_HEIGHT - 100:
            self.speed_y *= -1
    
    def draw(self, screen):
        # Draw glow
        for i in range(3, 0, -1):
            pygame.draw.circle(screen, (*self.color, 50), 
                              (int(self.x), int(self.y)), self.size + i * 2)
        
        # Draw bug body
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)
        
        # Draw eyes
        eye_size = self.size // 3
        pygame.draw.circle(screen, WHITE, (int(self.x - self.size//3), int(self.y - self.size//3)), eye_size)
        pygame.draw.circle(screen, WHITE, (int(self.x + self.size//3), int(self.y - self.size//3)), eye_size)
        pygame.draw.circle(screen, BLACK, (int(self.x - self.size//3), int(self.y - self.size//3)), eye_size//2)
        pygame.draw.circle(screen, BLACK, (int(self.x + self.size//3), int(self.y - self.size//3)), eye_size//2)
        
        # Show hits left if needed
        if self.hits_left > 1:
            font = pygame.font.Font(None, 20)
            text = font.render(str(self.hits_left), True, WHITE)
            screen.blit(text, (self.x - 10, self.y - 30))

class BugSmasherGame:
    def __init__(self, callback=None):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🐛 Bug Smasher - Debug Warrior")
        self.clock = pygame.time.Clock()
        self.callback = callback
        
        # Game state
        self.bugs = []
        self.particles = []
        self.score = 0
        self.bugs_smashed = 0
        self.streak = 0
        self.streak_timer = 0
        self.focus_energy = 0
        self.start_time = pygame.time.get_ticks()
        self.game_over = False
        self.show_summary = False
        self.current_tip = "💡 Click bugs to smash them!"
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Spawn timer
        self.spawn_delay = 1000
        self.last_spawn = pygame.time.get_ticks()
        
        # Game duration: 2 minutes
        self.game_duration = 120000
        
        # Simple sounds (using pygame's beep)
        try:
            pygame.mixer.init()
        except:
            pass
    
    def get_difficulty(self):
        """Return difficulty multiplier based on game progress"""
        elapsed = pygame.time.get_ticks() - self.start_time
        progress = elapsed / self.game_duration
        
        if progress < 0.25:
            return 1.5  # Slow start
        elif progress < 0.5:
            return 1.0  # Medium
        elif progress < 0.75:
            return 0.6  # Fast
        else:
            return 1.2  # Cooldown
    
    def spawn_bug(self):
        """Spawn a new bug"""
        if self.game_over or self.show_summary:
            return
        
        elapsed = pygame.time.get_ticks() - self.start_time
        progress = elapsed / self.game_duration
        
        # Choose bug type based on progress
        if progress < 0.3:
            bug_type = random.choice(['syntax', 'logic'])
        elif progress < 0.7:
            bug_type = random.choice(['syntax', 'logic', 'runtime'])
        else:
            bug_type = random.choice(['syntax', 'logic', 'runtime', 'memory'])
        
        # Spawn at edge
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            x = random.randint(50, SCREEN_WIDTH - 50)
            y = 50
        elif side == 'bottom':
            x = random.randint(50, SCREEN_WIDTH - 50)
            y = SCREEN_HEIGHT - 100
        elif side == 'left':
            x = 50
            y = random.randint(80, SCREEN_HEIGHT - 100)
        else:
            x = SCREEN_WIDTH - 50
            y = random.randint(80, SCREEN_HEIGHT - 100)
        
        bug = Bug(x, y, bug_type)
        self.bugs.append(bug)
    
    def smash_bug(self, pos):
        """Smash bug at mouse position"""
        mouse_x, mouse_y = pos
        
        for bug in self.bugs[:]:
            dx = bug.x - mouse_x
            dy = bug.y - mouse_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < bug.size:
                bug.hits_left -= 1
                
                if bug.hits_left <= 0:
                    # Bug smashed!
                    self.bugs.remove(bug)
                    
                    # Add particles
                    for _ in range(10):
                        self.particles.append(Particle(bug.x, bug.y, bug.color))
                    
                    # Add score
                    self.score += bug.points
                    self.bugs_smashed += 1
                    self.focus_energy = min(100, self.focus_energy + bug.points // 5)
                    
                    # Streak system
                    self.streak += 1
                    self.streak_timer = 1000
                    
                    # Streak bonuses
                    if self.streak == 3:
                        self.score += 15
                        self.current_tip = "⚡ QUICK DEBUG! +15 bonus!"
                    elif self.streak == 5:
                        self.score += 30
                        self.current_tip = "🔥 FLOW STATE! +30 bonus!"
                    
                    # Show tip every 5 bugs
                    if self.bugs_smashed % 5 == 0:
                        self.current_tip = bug.tip
                else:
                    # Hit but not destroyed
                    self.current_tip = f"Hit! {bug.hits_left} more hits!"
    
    def draw_background(self):
        """Draw cyberpunk background"""
        self.screen.fill(DARK_PURPLE)
        
        # Draw grid
        grid_spacing = 50
        for x in range(0, SCREEN_WIDTH, grid_spacing):
            pygame.draw.line(self.screen, (30, 20, 50), (x, 0), (x, SCREEN_HEIGHT), 1)
        for y in range(0, SCREEN_HEIGHT, grid_spacing):
            pygame.draw.line(self.screen, (30, 20, 50), (0, y), (SCREEN_WIDTH, y), 1)
    
    def draw_hud(self):
        """Draw HUD"""
        elapsed = pygame.time.get_ticks() - self.start_time
        time_left = max(0, (self.game_duration - elapsed) // 1000)
        minutes = time_left // 60
        seconds = time_left % 60
        
        # Progress bar
        pygame.draw.rect(self.screen, (40, 40, 60), (20, 20, 300, 20))
        progress_width = int(300 * (self.focus_energy / 100))
        pygame.draw.rect(self.screen, NEON_GREEN, (20, 20, progress_width, 20))
        
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (350, 20))
        
        # Streak
        streak_text = self.font_medium.render(f"Streak: {self.streak}x", True, NEON_PINK)
        self.screen.blit(streak_text, (550, 20))
        
        # Time
        time_color = RED if time_left < 10 else WHITE
        time_text = self.font_medium.render(f"Time: {minutes}:{seconds:02d}", True, time_color)
        self.screen.blit(time_text, (750, 20))
        
        # Focus text
        focus_text = self.font_small.render("FOCUS ENERGY", True, NEON_BLUE)
        self.screen.blit(focus_text, (20, 45))
        
        # Tip
        tip_bg = pygame.Surface((SCREEN_WIDTH - 40, 40))
        tip_bg.set_alpha(180)
        tip_bg.fill(BLACK)
        self.screen.blit(tip_bg, (20, SCREEN_HEIGHT - 60))
        
        tip_text = self.font_small.render(self.current_tip, True, NEON_BLUE)
        self.screen.blit(tip_text, (30, SCREEN_HEIGHT - 50))
    
    def draw_summary(self):
        """Draw summary screen"""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        y = 200
        
        # Title
        title = self.font_large.render("DEBUG SESSION COMPLETE", True, NEON_BLUE)
        self.screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, y))
        y += 80
        
        # Stats
        stats = [
            f"Bugs Smashed: {self.bugs_smashed}",
            f"Focus Energy: {self.focus_energy}%",
            f"Highest Streak: {self.streak}",
            f"Total Score: {self.score}"
        ]
        
        for stat in stats:
            text = self.font_medium.render(stat, True, WHITE)
            self.screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, y))
            y += 50
        
        y += 30
        
        # Productivity tip
        tips = [
            "💡 Take a 5-min break when stuck!",
            "💡 Break problems into smaller functions",
            "💡 Use print statements to debug",
            "💡 Google is your friend!",
            "💡 Rubber duck debugging works!"
        ]
        tip_text = self.font_small.render(random.choice(tips), True, NEON_GREEN)
        self.screen.blit(tip_text, (SCREEN_WIDTH//2 - tip_text.get_width()//2, y))
        y += 60
        
        # Boost message
        boost_text = self.font_medium.render(f"✨ FOCUS BOOST: +{self.focus_energy} points! ✨", 
                                             True, NEON_PINK)
        self.screen.blit(boost_text, (SCREEN_WIDTH//2 - boost_text.get_width()//2, y))
        y += 80
        
        # Buttons
        return_btn = pygame.Rect(SCREEN_WIDTH//2 - 110, y, 100, 40)
        again_btn = pygame.Rect(SCREEN_WIDTH//2 + 10, y, 100, 40)
        
        pygame.draw.rect(self.screen, NEON_GREEN, return_btn)
        pygame.draw.rect(self.screen, NEON_BLUE, again_btn)
        
        return_text = self.font_small.render("RETURN", True, BLACK)
        again_text = self.font_small.render("AGAIN", True, BLACK)
        
        self.screen.blit(return_text, (return_btn.x + 25, return_btn.y + 12))
        self.screen.blit(again_text, (again_btn.x + 25, again_btn.y + 12))
        
        return return_btn, again_btn
    
    def run(self):
        """Main game loop"""
        running = True
        return_btn = None
        again_btn = None
        
        while running:
            current_time = pygame.time.get_ticks()
            elapsed = current_time - self.start_time
            
            # Check game end
            if not self.game_over and elapsed >= self.game_duration:
                self.game_over = True
                self.show_summary = True
            
            # Update streak timer
            if self.streak_timer > 0:
                self.streak_timer -= self.clock.get_time()
                if self.streak_timer <= 0:
                    self.streak = 0
            
            # Spawn bugs
            if not self.game_over and not self.show_summary:
                delay = int(self.spawn_delay * self.get_difficulty())
                if current_time - self.last_spawn > delay:
                    self.spawn_bug()
                    self.last_spawn = current_time
            
            # Update bugs
            for bug in self.bugs:
                bug.update()
            
            # Update particles
            self.particles = [p for p in self.particles if p.update()]
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.show_summary:
                        if return_btn and return_btn.collidepoint(event.pos):
                            if self.callback:
                                self.callback({
                                    'score': self.score,
                                    'bugs_smashed': self.bugs_smashed,
                                    'focus_energy': self.focus_energy
                                })
                            running = False
                        elif again_btn and again_btn.collidepoint(event.pos):
                            return self.restart()
                    else:
                        self.smash_bug(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.show_summary:
                        self.smash_bug(pygame.mouse.get_pos())
            
            # Draw everything
            self.draw_background()
            
            for bug in self.bugs:
                bug.draw(self.screen)
            
            for particle in self.particles:
                particle.draw(self.screen)
            
            self.draw_hud()
            
            if self.show_summary:
                return_btn, again_btn = self.draw_summary()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        return None
    
    def restart(self):
        """Restart the game"""
        return BugSmasherGame(self.callback).run()


def launch_game(callback=None):
    """Launch the game from your main app"""
    game = BugSmasherGame(callback)
    result = game.run()
    return result


if __name__ == "__main__":
    # Test the game standalone
    def test_callback(results):
        print(f"\nGame Results: {results}")
        print(f"Points to add: +{results['focus_energy']}")
    
    print("Starting Bug Smasher Game...")
    launch_game(test_callback)