# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import logging

# Configure matplotlib to use a specific font and suppress debug messages
plt.rcParams['font.family'] = 'DejaVu Sans'
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Rectangle
from kivy.uix.gridlayout import GridLayout

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Set default font for Kivy
from kivy.resources import resource_add_path
resource_add_path('C:/Windows/Fonts')
Config.set('kivy', 'default_font', ['DejaVuSans'])

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(6,3,0.9)
action2rotation = [0,5,-5]
last_reward = 0
scores = []
best_accuracy = 0
epoch_count = 0
MAX_EPOCHS = 2  # Changed from 50 to 2

# Load the ideal path mask
ideal_path = CoreImage("./images/MASK1.png")

# Initializing the map
first_update = True
last_distance = float('inf')  # Initialize last_distance with infinity

def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global swap
    global last_distance
    global longueur, largeur
    
    # Initialize sand array with window dimensions
    longueur = 1429  # Window width
    largeur = 660    # Window height
    sand = np.zeros((longueur, largeur))
    
    # Load and process the mask image
    img = PILImage.open("./images/MASK1.png").convert('L')
    img = img.resize((longueur, largeur))
    sand = np.asarray(img)/255
    
    # Load citymap image
    citymap = PILImage.open("./images/citymap.png")
    citymap = citymap.resize((longueur, largeur))
    citymap = citymap.convert('RGB')  # Convert to RGB to allow colored drawing
    
    # Create a copy of the mask image for drawing
    mask_img = img.copy()
    mask_img = mask_img.convert('RGB')  # Convert to RGB to allow colored drawing
    
    # Define target points with their coordinates
    targets = {
        1: {'x': 42, 'y': 38, 'label': 'A1'},
        2: {'x': 56, 'y': 44, 'label': 'A2'},
        3: {'x': 49, 'y': 26, 'label': 'A3'}
    }
    
    # Draw target points on both images
    for target in targets.values():
        x, y = target['x'] * 20, target['y'] * 20  # Scale coordinates
        
        # Draw outer red circle
        for dx in range(-35, 36):
            for dy in range(-35, 36):
                if dx*dx + dy*dy <= 1225:  # Circle with radius 35
                    px, py = x + dx, y + dy
                    if 0 <= px < longueur and 0 <= py < largeur:
                        # Draw on citymap
                        citymap.putpixel((px, py), (255, 0, 0))  # Red color
                        # Draw on mask
                        mask_img.putpixel((px, py), (255, 0, 0))  # Red color
        
        # Draw middle white circle
        for dx in range(-30, 31):
            for dy in range(-30, 31):
                if dx*dx + dy*dy <= 900:  # Circle with radius 30
                    px, py = x + dx, y + dy
                    if 0 <= px < longueur and 0 <= py < largeur:
                        # Draw on citymap
                        citymap.putpixel((px, py), (255, 255, 255))  # White color
                        # Draw on mask
                        mask_img.putpixel((px, py), (255, 255, 255))  # White color
        
        # Draw inner red circle
        for dx in range(-25, 26):
            for dy in range(-25, 26):
                if dx*dx + dy*dy <= 625:  # Circle with radius 25
                    px, py = x + dx, y + dy
                    if 0 <= px < longueur and 0 <= py < largeur:
                        # Draw on citymap
                        citymap.putpixel((px, py), (255, 0, 0))  # Red color
                        # Draw on mask
                        mask_img.putpixel((px, py), (255, 0, 0))  # Red color
    
    # Save both modified images
    citymap.save("./images/citymap.png")
    mask_img.save("./images/MASK1.png")
    
    # Ensure goal coordinates are within bounds
    goal_x = min(1420, longueur - 1)  # Start with Target A1, but ensure it's within bounds
    goal_y = min(622, largeur - 1)
    
    first_update = False
    swap = 0  # Initialize swap to 0 for first target
    last_distance = float('inf')  # Reset last_distance when initializing


# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)
    boundary_awareness = NumericProperty(0)
    speed = NumericProperty(0)  # New property for speed
    max_speed = NumericProperty(5)  # Maximum speed limit
    min_speed = NumericProperty(0.5)  # Minimum speed limit

    def move(self, rotation):
        # Update car's position based on current velocity
        self.pos = Vector(*self.velocity) + self.pos
        
        # Update rotation and angle
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        
        # Update sensors
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        
        # Add boundary checks before accessing sand array
        def safe_sum(x, y):
            x = max(0, min(int(x), longueur-1))
            y = max(0, min(int(y), largeur-1))
            x_start = max(0, x-10)
            x_end = min(longueur, x+10)
            y_start = max(0, y-10)
            y_end = min(largeur, y+10)
            return int(np.sum(sand[x_start:x_end, y_start:y_end]))/200.
        
        # Calculate signals with safe boundary handling
        self.signal1 = safe_sum(self.sensor1_x, self.sensor1_y)
        self.signal2 = safe_sum(self.sensor2_x, self.sensor2_y)
        self.signal3 = safe_sum(self.sensor3_x, self.sensor3_y)
        
        # Calculate boundary awareness as average of all sensors
        self.boundary_awareness = (self.signal1 + self.signal2 + self.signal3) / 3
        
        # Handle boundary cases with stronger signals
        if self.sensor1_x > longueur-10 or self.sensor1_x < 10 or self.sensor1_y > largeur-10 or self.sensor1_y < 10:
            self.signal1 = 20.  # Increased boundary signal
        if self.sensor2_x > longueur-10 or self.sensor2_x < 10 or self.sensor2_y > largeur-10 or self.sensor2_y < 10:
            self.signal2 = 20.  # Increased boundary signal
        if self.sensor3_x > longueur-10 or self.sensor3_x < 10 or self.sensor3_y > largeur-10 or self.sensor3_y < 10:
            self.signal3 = 20.  # Increased boundary signal

    def set_speed(self, speed_value):
        """Set the car's speed while maintaining current direction"""
        self.speed = max(self.min_speed, min(speed_value, self.max_speed))
        self.velocity = Vector(self.speed, 0).rotate(self.angle)
        self.velocity_x = self.velocity[0]
        self.velocity_y = self.velocity[1]

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

class Game(Widget):
    """Main game widget containing the car and game logic"""
    
    # Widget properties
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    label1 = ObjectProperty(None)
    label2 = ObjectProperty(None)
    label3 = ObjectProperty(None)
    score_label = ObjectProperty(None)
    stats_label = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Game, self).__init__(**kwargs)
        # Initialize game state
        self.current_target = 1
        self.targets = {
            1: {'x': 42, 'y': 38, 'label': 'A1', 'reward': 10},
            2: {'x': 56, 'y': 44, 'label': 'A2', 'reward': 15},
            3: {'x': 49, 'y': 26, 'label': 'A3', 'reward': 20}
        }
        self.target_reached = False
        self.target_threshold = 5
        self.total_rewards = 0
        self.episode_count = 0
        self.best_score = 0
        self.last_distance = float('inf')
        self.stats = {
            'targets_reached': 0,
            'collisions': 0,
            'avg_speed': 0,
            'total_distance': 0,
            'checkpoints_reached': 0,
            'time_spent': 0,
            'best_lap_time': float('inf')
        }
        self.paused = False
        self.manual_control = False
        self.game_mode = 'training'  # 'training' or 'testing'
        self.difficulty = 'normal'    # 'easy', 'normal', 'hard'
        self.checkpoints = []         # List to store checkpoint positions
        self.current_checkpoint = 0
        self.start_time = time.time()
        self.lap_start_time = time.time()
        self.keyboard = Window.request_keyboard(self._on_keyboard_down, self)
        self.keyboard.bind(on_key_down=self._on_keyboard_down)
        self.keyboard.bind(on_key_up=self._on_keyboard_up)
        
        # Draw target points with larger, more visible circles
        with self.canvas:
            for target in self.targets.values():
                # Draw outer red circle (larger)
                Color(1, 0, 0, 1)  # Red color
                Ellipse(pos=(target['x']*20 - 35, target['y']*20 - 35), size=(70, 70))
                
                # Draw middle white circle
                Color(1, 1, 1, 1)  # White color
                Ellipse(pos=(target['x']*20 - 30, target['y']*20 - 30), size=(60, 60))
                
                # Draw inner red circle
                Color(1, 0, 0, 1)  # Red color
                Ellipse(pos=(target['x']*20 - 25, target['y']*20 - 25), size=(50, 50))
                
                # Draw target label with larger font and better positioning
                Label(
                    text=target['label'],
                    pos=(target['x']*20 + 40, target['y']*20 - 15),
                    color=(1, 0, 0, 1),
                    font_size='28sp',
                    bold=True
                ).texture  # Force texture creation

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        """Handle keyboard input"""
        if keycode[1] == 'spacebar':
            self.paused = not self.paused
            print(f"Game {'Paused' if self.paused else 'Resumed'}")
        elif keycode[1] == 'm':
            self.manual_control = not self.manual_control
            print(f"Mode: {'Manual' if self.manual_control else 'AI'} Control")
        elif keycode[1] == 'c':  # Add checkpoint
            self._add_checkpoint()
        elif self.manual_control:
            if keycode[1] == 'left':
                self.car.move(-5)
            elif keycode[1] == 'right':
                self.car.move(5)
            elif keycode[1] == 'up':
                self.car.set_speed(2.0)
                if hasattr(self, 'speed_label') and self.speed_label:
                    self.speed_label.text = f'Current Speed: 2.0'
            elif keycode[1] == 'down':
                self.car.set_speed(0.5)
                if hasattr(self, 'speed_label') and self.speed_label:
                    self.speed_label.text = f'Current Speed: 0.5'

    def _on_keyboard_up(self, keyboard, keycode):
        """Handle key release events"""
        if self.manual_control:
            if keycode[1] in ['up', 'down']:
                # Reset to normal speed when up/down keys are released
                self.car.set_speed(1.0)
                if hasattr(self, 'speed_label') and self.speed_label:
                    self.speed_label.text = f'Current Speed: 1.0'

    def _add_checkpoint(self):
        """Add a checkpoint at current car position"""
        if len(self.checkpoints) < 5:  # Limit to 5 checkpoints
            self.checkpoints.append({
                'x': self.car.x,
                'y': self.car.y,
                'label': f'CP{len(self.checkpoints) + 1}'
            })
            print(f"Checkpoint {len(self.checkpoints)} added!")

    def _check_checkpoint(self):
        """Check if car has reached current checkpoint"""
        if self.current_checkpoint < len(self.checkpoints):
            checkpoint = self.checkpoints[self.current_checkpoint]
            distance = np.sqrt((self.car.x - checkpoint['x'])**2 + (self.car.y - checkpoint['y'])**2)
            if distance < self.target_threshold:
                self.stats['checkpoints_reached'] += 1
                self.current_checkpoint += 1
                print(f"Checkpoint {self.current_checkpoint} reached!")
                return True
        return False

    def update(self, dt):
        """Main game update loop"""
        if self.paused:
            return

        global brain, last_reward, scores, goal_x, goal_y, longueur, largeur, swap, epoch_count, best_accuracy

        # Update time statistics
        self.stats['time_spent'] = time.time() - self.start_time

        # Initialize game if first update
        if first_update:
            init()
            print("\n=== Starting Training ===")
            print(f"Training will run for {MAX_EPOCHS} epochs")
            print("========================\n")
            # Initialize epoch progress bar
            if hasattr(self, 'epoch_progress_bar'):
                self.epoch_progress_bar.size_hint_x = 0

        # Update current target
        current_target = self.targets[self.current_target]
        goal_x = current_target['x']
        goal_y = current_target['y']

        # Calculate orientation to target
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        
        # Prepare state signal for AI
        last_signal = [
            self.car.signal1, 
            self.car.signal2, 
            self.car.signal3, 
            orientation, 
            -orientation, 
            self.car.boundary_awareness
        ]
        
        # Get AI action and update car
        if not self.manual_control and self.game_mode == 'training':
            action = brain.update(last_reward, last_signal)
            scores.append(brain.score())
            rotation = action2rotation[action]
            self.car.move(rotation)
            
            # Check if episode is complete (all targets reached)
            if self.current_target == 1 and self.target_reached:
                epoch_count += 1
                current_accuracy = brain.score()  # Get current accuracy
                
                # Update epoch progress visualization
                if hasattr(self, 'epoch_progress_bar'):
                    progress = epoch_count / MAX_EPOCHS
                    self.epoch_progress_bar.size_hint_x = progress
                    self.epoch_progress_label.text = f'Epoch Progress: {epoch_count}/{MAX_EPOCHS}'
                
                print(f"\n=== Epoch {epoch_count}/{MAX_EPOCHS} Complete ===")
                print(f"Current Accuracy: {current_accuracy:.2f}")
                print(f"Best Accuracy: {best_accuracy:.2f}")
                print(f"Total Rewards: {self.total_rewards:.2f}")
                print(f"Targets Reached: {self.stats['targets_reached']}")
                print(f"Average Speed: {self.stats['avg_speed']:.2f}")
                print("========================\n")
                
                # Save best model
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    brain.save()
                    print(f"New best accuracy: {best_accuracy:.2f}")
                    print("Model saved to 'last_brain.pth'")
                
                # Check if training is complete
                if epoch_count >= MAX_EPOCHS:
                    print("\n=== Training Complete ===")
                    print(f"Final Best Accuracy: {best_accuracy:.2f}")
                    print(f"Total Episodes: {self.episode_count}")
                    print(f"Final Score: {self.total_rewards:.2f}")
                    print(f"Best Lap Time: {self.stats['best_lap_time']:.2f}s")
                    print("========================\n")
                    self.game_mode = 'testing'  # Switch to testing mode
                    return
        
        # Update car position and sensors
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        # Update statistics
        self._update_stats()

        # Handle car-environment interaction
        self._handle_car_environment_interaction(distance, current_target)

        # Check checkpoint
        self._check_checkpoint()

        # Check if all targets are reached (lap completed)
        if self.current_target == 1 and self.target_reached:
            lap_time = time.time() - self.lap_start_time
            if lap_time < self.stats['best_lap_time']:
                self.stats['best_lap_time'] = lap_time
                print(f"New best lap time: {lap_time:.2f} seconds!")
            self.lap_start_time = time.time()
            self.checkpoints = []  # Reset checkpoints for new lap
            self.current_checkpoint = 0

    def _update_stats(self):
        """Update game statistics"""
        # Update average speed
        self.stats['avg_speed'] = (self.stats['avg_speed'] * self.episode_count + self.car.speed) / (self.episode_count + 1)
        
        # Update total distance
        if hasattr(self, 'last_pos'):
            dx = self.car.x - self.last_pos[0]
            dy = self.car.y - self.last_pos[1]
            self.stats['total_distance'] += np.sqrt(dx*dx + dy*dy)
        self.last_pos = (self.car.x, self.car.y)
        
        # Update display
        self._update_display()

    def _update_display(self):
        """Update the display with current stats"""
        # Update score label
        if hasattr(self, 'score_label') and self.score_label:
            self.score_label.text = f'🎯 Score: {self.total_rewards:.1f}'
        
        # Update episode label
        if hasattr(self, 'episode_label') and self.episode_label:
            self.episode_label.text = f'🔄 Episode: {self.episode_count}'
        
        # Update target label
        if hasattr(self, 'target_label') and self.target_label:
            current_target = self.targets[self.current_target]
            self.target_label.text = f'🎯 Target: {current_target["label"]}'
            self.target_label.color = (1, 0, 0, 1)
        
        # Update speed label
        if hasattr(self, 'speed_label') and self.speed_label:
            current_speed = float(np.sqrt(self.car.velocity_x**2 + self.car.velocity_y**2))
            self.speed_label.text = f'⚡ Speed: {current_speed:.1f}'
        
        # Update distance label
        if hasattr(self, 'distance_label') and self.distance_label:
            current_target = self.targets[self.current_target]
            distance = np.sqrt((self.car.x - current_target['x'])**2 + (self.car.y - current_target['y'])**2)
            self.distance_label.text = f'📏 Distance: {distance:.1f}'

        # Update stats label with additional information
        if hasattr(self, 'stats_label') and self.stats_label:
            stats_text = (
                f'Mode: {self.game_mode.title()}\n'
                f'Difficulty: {self.difficulty.title()}\n'
                f'Checkpoints: {self.stats["checkpoints_reached"]}/{len(self.checkpoints)}\n'
                f'Best Lap: {self.stats["best_lap_time"]:.1f}s\n'
                f'Time: {self.stats["time_spent"]:.1f}s'
            )
            self.stats_label.text = stats_text

    def _handle_car_environment_interaction(self, distance, current_target):
        """Handle car's interaction with environment and targets"""
        global last_reward, longueur, largeur
        
        # Ensure car coordinates are within valid bounds
        car_x = max(0, min(int(self.car.x), longueur-1))
        car_y = max(0, min(int(self.car.y), largeur-1))
        
        if 0 <= car_x < longueur and 0 <= car_y < largeur:
            try:
                sand_value = sand[car_x, car_y]
                if sand_value < 0.5:  # Changed condition to check for white lines
                    self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                    last_reward = -2  # Penalty for going off the white lines
                    self.total_rewards += last_reward  # Update total rewards immediately
                else:
                    self.car.velocity = Vector(2, 0).rotate(self.car.angle)
                    last_reward = -0.1  # Small penalty for normal driving
                    self.total_rewards += last_reward  # Update total rewards immediately
                    if distance < self.last_distance:
                        last_reward = 0.2  # Reward for getting closer to goal
                        self.total_rewards += last_reward  # Update total rewards immediately
            except IndexError:
                # Handle index error
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                last_reward = -2
                self.total_rewards += last_reward  # Update total rewards immediately
                self.car.x = max(0, min(self.car.x, longueur-1))
                self.car.y = max(0, min(self.car.y, largeur-1))
        else:
            # If car is out of bounds, slow it down and give penalty
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            last_reward = -2  # Penalty for going out of bounds
            self.total_rewards += last_reward  # Update total rewards immediately
            
            # Reset car position to last valid position
            self.car.x = max(0, min(self.car.x, longueur-1))
            self.car.y = max(0, min(self.car.y, largeur-1))

        # Check target reached
        if distance < self.target_threshold and not self.target_reached:
            self._handle_target_reached(current_target)

        # Handle boundary collisions
        self._handle_boundary_collisions()

        self.last_distance = distance  # Update last_distance at the end of the method

    def _handle_target_reached(self, current_target):
        """Handle logic when a target is reached"""
        self.target_reached = True
        last_reward = current_target['reward']  # Dynamic reward based on target
        self.total_rewards += last_reward  # Update total rewards immediately
        
        # Update best score
        if self.total_rewards > self.best_score:
            self.best_score = self.total_rewards
            print(f"New best score: {self.best_score}")
        
        # Move to next target
        self.current_target = (self.current_target % 3) + 1
        self.target_reached = False
        
        print(f"Target {current_target['label']} reached! Moving to {self.targets[self.current_target]['label']}")
        print(f"Current total rewards: {self.total_rewards}")

    def _handle_boundary_collisions(self):
        """Handle car collisions with boundaries"""
        global last_reward
        
        if self.car.x < 5:
            self.car.x = 5
            last_reward = -2
            self.total_rewards += last_reward  # Update total rewards immediately
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            last_reward = -2
            self.total_rewards += last_reward  # Update total rewards immediately
        if self.car.y < 5:
            self.car.y = 5
            last_reward = -2
            self.total_rewards += last_reward  # Update total rewards immediately
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            last_reward = -2
            self.total_rewards += last_reward  # Update total rewards immediately

    def serve_car(self):
        """Initialize car position and state"""
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)
        self.car.angle = 0  # Reset car angle
        self.car.set_speed(1.0)  # Set initial speed to normal
        self.episode_count += 1
        self.last_distance = float('inf')  # Reset last_distance when serving car
        print(f"Episode {self.episode_count} started")

# Adding the painting tools

class MyPaintWidget(Widget):
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            
            # Ensure coordinates are within bounds
            x = max(0, min(int(touch.x), longueur-1))
            y = max(0, min(int(touch.y), largeur-1))
            sand[x, y] = 1
            
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = max(0, min(int(touch.x), longueur-1))
            y = max(0, min(int(touch.y), largeur-1))
            
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            
            # Ensure coordinates are within bounds for sand array
            x_start = max(0, x - 10)
            x_end = min(longueur, x + 10)
            y_start = max(0, y - 10)
            y_end = min(largeur, y + 10)
            sand[x_start:x_end, y_start:y_end] = 1
            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        
        # Create main container for all UI elements
        main_container = BoxLayout(
            orientation='vertical',
            size_hint=(None, None),
            size=(300, 400),  # Increased from (200, 300)
            pos_hint={'right': 1, 'top': 1},
            padding=5,
            spacing=5
        )
        
        # Create header section with status indicators
        header = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=50,  # Increased from 35
            spacing=2
        )
        
        # Create title with modern styling
        info_title = Label(
            text='Self-Driving Car Dashboard',
            color=(0.2, 0.8, 1, 1),
            font_size='16sp',  # Increased from 12sp
            bold=True,
            size_hint_y=None,
            height=25,
        )
        
        # Create status indicators container
        status_container = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=20,
            spacing=5
        )
        
        # Create mode indicator
        parent.mode_label = Label(
            text='AI Control',
            color=(0.2, 0.8, 1, 1),
            font_size='14sp',
            bold=True
        )
        
        # Create pause indicator
        parent.pause_label = Label(
            text='Running',
            color=(0.2, 0.8, 0.2, 1),
            font_size='14sp',
            bold=True
        )
        
        status_container.add_widget(parent.mode_label)
        status_container.add_widget(parent.pause_label)
        
        header.add_widget(info_title)
        header.add_widget(status_container)
        main_container.add_widget(header)
        
        # Create stats section
        stats_section = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=120,
            spacing=2,
            padding=[0, 2]
        )
        
        # Create section title
        stats_title = Label(
            text='Game Statistics',
            color=(0.2, 0.8, 1, 1),
            font_size='14sp',
            bold=True,
            size_hint_y=None,
            height=25
        )
        
        # Create epoch progress section
        epoch_progress = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=40,
            spacing=2
        )
        
        # Create epoch progress bar background
        progress_bg = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=15,
            padding=1
        )
        progress_bg.canvas.before.add(Color(0.2, 0.2, 0.2, 1))
        progress_bg.canvas.before.add(Rectangle(pos=progress_bg.pos, size=progress_bg.size))
        
        # Create epoch progress bar
        parent.epoch_progress_bar = BoxLayout(
            orientation='horizontal',
            size_hint_x=0,
            size_hint_y=None,
            height=13
        )
        parent.epoch_progress_bar.canvas.before.add(Color(0.2, 0.8, 1, 1))
        parent.epoch_progress_bar.canvas.before.add(Rectangle(pos=parent.epoch_progress_bar.pos, size=parent.epoch_progress_bar.size))
        
        # Create epoch progress label
        parent.epoch_progress_label = Label(
            text='Epoch Progress: 0/2',
            color=(0.2, 0.8, 1, 1),
            font_size='12sp',
            size_hint_y=None,
            height=20
        )
        
        progress_bg.add_widget(parent.epoch_progress_bar)
        epoch_progress.add_widget(parent.epoch_progress_label)
        epoch_progress.add_widget(progress_bg)
        
        # Create stats grid
        stats_grid = GridLayout(
            cols=2,
            spacing=2,
            size_hint_y=None,
            height=90
        )
        
        # Create labels with improved styling
        parent.score_label = Label(
            text='Score: 0.0',
            color=(0.2, 0.8, 1, 1),
            font_size='14sp',
            bold=True,
            size_hint_y=None,
            height=25
        )
        parent.episode_label = Label(
            text='Episode: 0',
            color=(0.8, 0.8, 0.8, 1),
            font_size='14sp',
            size_hint_y=None,
            height=25
        )
        parent.target_label = Label(
            text='Target: A1',
            color=(0.2, 0.8, 0.2, 1),
            font_size='14sp',
            bold=True,
            size_hint_y=None,
            height=25
        )
        parent.speed_label = Label(
            text='Speed: 1.0',
            color=(0.8, 0.2, 0.2, 1),
            font_size='14sp',
            bold=True,
            size_hint_y=None,
            height=25
        )
        parent.distance_label = Label(
            text='Distance: 0.0',
            color=(0.2, 0.8, 0.2, 1),
            font_size='14sp',
            size_hint_y=None,
            height=25
        )
        
        stats_grid.add_widget(parent.score_label)
        stats_grid.add_widget(parent.episode_label)
        stats_grid.add_widget(parent.target_label)
        stats_grid.add_widget(parent.speed_label)
        stats_grid.add_widget(parent.distance_label)
        
        stats_section.add_widget(stats_title)
        stats_section.add_widget(epoch_progress)
        stats_section.add_widget(stats_grid)
        main_container.add_widget(stats_section)
        
        # Create speed control section
        speed_section = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=80,
            spacing=2
        )
        
        # Create speed title
        speed_title = Label(
            text='Speed Control',
            color=(0.2, 0.8, 1, 1),
            font_size='14sp',
            bold=True,
            size_hint_y=None,
            height=25
        )
        
        # Create speed buttons
        speed_buttons = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=40,
            spacing=5,
            padding=[0, 2]
        )
        
        def create_speed_button(text, color):
            btn = Button(
                text=text,
                size_hint_x=None,
                width=80,
                background_color=color,
                font_size='14sp',
                bold=True
            )
            btn.bind(on_press=lambda x: setattr(btn, 'background_color', (color[0]*0.8, color[1]*0.8, color[2]*0.8, 1)))
            btn.bind(on_release=lambda x: setattr(btn, 'background_color', color))
            return btn
        
        slow_btn = create_speed_button('Slow', (0.8, 0.2, 0.2, 1))
        normal_btn = create_speed_button('Normal', (0.2, 0.8, 0.2, 1))
        fast_btn = create_speed_button('Fast', (0.2, 0.2, 0.8, 1))
        
        speed_buttons.add_widget(slow_btn)
        speed_buttons.add_widget(normal_btn)
        speed_buttons.add_widget(fast_btn)
        
        speed_section.add_widget(speed_title)
        speed_section.add_widget(speed_buttons)
        main_container.add_widget(speed_section)
        
        # Create action buttons section
        action_section = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=80,
            spacing=2
        )
        
        # Create action title
        action_title = Label(
            text='Game Actions',
            color=(0.2, 0.8, 1, 1),
            font_size='14sp',
            bold=True,
            size_hint_y=None,
            height=25
        )
        
        action_buttons = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=40,
            spacing=5
        )
        
        def create_action_button(text, color):
            btn = Button(
                text=text,
                size_hint_x=None,
                width=80,
                background_color=color,
                font_size='14sp',
                bold=True
            )
            btn.bind(on_press=lambda x: setattr(btn, 'background_color', (color[0]*0.8, color[1]*0.8, color[2]*0.8, 1)))
            btn.bind(on_release=lambda x: setattr(btn, 'background_color', color))
            return btn
        
        clearbtn = create_action_button('Clear', (0.8, 0.2, 0.2, 1))
        savebtn = create_action_button('Save', (0.2, 0.8, 0.2, 1))
        loadbtn = create_action_button('Load', (0.2, 0.2, 0.8, 1))
        
        action_buttons.add_widget(clearbtn)
        action_buttons.add_widget(savebtn)
        action_buttons.add_widget(loadbtn)
        
        action_section.add_widget(action_title)
        action_section.add_widget(action_buttons)
        main_container.add_widget(action_section)
        
        # Create controls info section
        controls_section = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=80,
            spacing=2
        )
        
        # Create controls title
        controls_title = Label(
            text='Keyboard Controls',
            color=(0.2, 0.8, 1, 1),
            font_size='14sp',
            bold=True,
            size_hint_y=None,
            height=25
        )
        
        controls_grid = GridLayout(
            cols=2,
            spacing=2,
            size_hint_y=None,
            height=50
        )
        
        controls = [
            ('Space', 'Pause/Resume'),
            ('M', 'Toggle AI/Manual'),
            ('C', 'Add Checkpoint'),
            ('↑↓', 'Speed Control'),
            ('←→', 'Rotate')
        ]
        
        for key, action in controls:
            key_label = Label(
                text=key,
                color=(0.2, 0.8, 1, 1),
                font_size='14sp',
                bold=True,
                size_hint_y=None,
                height=25
            )
            action_label = Label(
                text=action,
                color=(0.8, 0.8, 0.8, 1),
                font_size='14sp',
                size_hint_y=None,
                height=25
            )
            controls_grid.add_widget(key_label)
            controls_grid.add_widget(action_label)
        
        controls_section.add_widget(controls_title)
        controls_section.add_widget(controls_grid)
        main_container.add_widget(controls_section)
        
        # Bind button actions with enhanced visual feedback
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        
        # Bind speed control actions with enhanced visual feedback
        def set_slow(obj):
            parent.car.set_speed(0.5)
            parent.speed_label.text = f'⚡ Speed: 0.5'
            parent.speed_label.color = (0.8, 0.2, 0.2, 1)
            parent.speed_label.bold = True
        
        def set_normal(obj):
            parent.car.set_speed(1.0)
            parent.speed_label.text = f'⚡ Speed: 1.0'
            parent.speed_label.color = (0.2, 0.8, 0.2, 1)
            parent.speed_label.bold = True
        
        def set_fast(obj):
            parent.car.set_speed(2.0)
            parent.speed_label.text = f'⚡ Speed: 2.0'
            parent.speed_label.color = (0.2, 0.2, 0.8, 1)
            parent.speed_label.bold = True
        
        slow_btn.bind(on_release=set_slow)
        normal_btn.bind(on_release=set_normal)
        fast_btn.bind(on_release=set_fast)
        
        # Add widgets to parent
        parent.add_widget(self.painter)
        parent.add_widget(main_container)
        
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
