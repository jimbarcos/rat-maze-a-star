import pygame
import random
import math
import heapq
import sys
from enum import Enum
from typing import List, Tuple, Dict, Set, Optional

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 780
CELL_SIZE = 40
GRID_WIDTH = 20
GRID_HEIGHT = 15
SIDEBAR_WIDTH = 300
MAZE_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
MAZE_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE
BOTTOM_PANEL_HEIGHT = HEIGHT - MAZE_AREA_HEIGHT
FPS = 60
PATH_ANIMATION_SPEED = 8  # Frames per step in path animation
SEARCH_ANIMATION_SPEED = 4  # Frames per step in search animation - can be modified by user
MIN_ANIMATION_SPEED = 1
MAX_ANIMATION_SPEED = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
BROWN = (139, 69, 19)
DARK_BROWN = (101, 67, 33)
CHEESE_COLOR = (255, 215, 0)  # Gold
OPEN_NODE_COLOR = (173, 216, 230)  # Light blue
CLOSED_NODE_COLOR = (255, 182, 193)  # Light pink
MOUSE_COLOR = (150, 150, 150)  # Lighter gray
MOUSE_EAR_COLOR = (200, 150, 150)  # Pink ears
F_SCORE_COLOR = (255, 255, 200)  # Light yellow
G_SCORE_COLOR = (200, 255, 200)  # Light green
PANEL_COLOR = (240, 240, 240)  # Light gray for UI panel
CURRENT_NODE_COLOR = (255, 140, 0)  # Orange for current node being processed
H_SCORE_COLOR = (255, 200, 255)  # Light purple for h-score
F_SCORE_INDICATOR_COLOR = (255, 240, 180, 180)  # Semi-transparent yellow

# Cell types
class CellType(Enum):
    EMPTY = 0
    WALL = 1
    START = 2
    GOAL = 3

# Define fonts
FONT_TINY = pygame.font.SysFont("Arial", 10)
FONT_SMALL = pygame.font.SysFont("Arial", 12)
FONT_MEDIUM = pygame.font.SysFont("Arial", 16)
FONT_LARGE = pygame.font.SysFont("Arial", 20, bold=True)
FONT_TITLE = pygame.font.SysFont("Arial", 24, bold=True)
FONT_SCORE = pygame.font.SysFont("Courier New", 11, bold=True)  # Smaller, monospaced font for scores
FONT_SCORE_MEDIUM = pygame.font.SysFont("Courier New", 12, bold=True)  # Medium score font
FONT_SCORE_LARGE = pygame.font.SysFont("Courier New", 14, bold=True)  # Large score font

# Game state
class GameState:
    def __init__(self):
        self.maze = [[CellType.EMPTY for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.start_pos = (1, 1)
        self.goal_pos = (GRID_WIDTH - 2, GRID_HEIGHT - 2)
        self.path = []
        self.came_from = {}
        self.open_set = set()
        self.closed_set = set()
        self.g_score = {}
        self.f_score = {}
        self.h_score = {}  # Add h_score dictionary
        self.show_maze = True
        self.show_graph = False
        self.show_path = True
        self.show_scores = True  # Master toggle for all scores
        self.show_f_score = True  # Submenu toggle for F score
        self.show_g_score = True  # Submenu toggle for G score
        self.show_h_score = True  # Submenu toggle for H score
        self.mouse_pos = self.start_pos
        self.animation_step = 0
        self.animation_counter = 0
        self.animation_active = False
        self.search_animation_active = False
        self.search_animation_step = 0
        self.search_path = []
        self.search_animation_speed = SEARCH_ANIMATION_SPEED
        self.current_node = None  # Current node being processed
        self.paused = False  # Flag to track pause state
        self.mouse_img = None
        self.cheese_img = None
        self.load_images()
        self.generate_maze()
    
    def load_images(self):
        # Create mouse image
        mouse_img = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        
        # Draw mouse body (gray ellipse)
        pygame.draw.ellipse(mouse_img, MOUSE_COLOR, (2, 2, CELL_SIZE-4, CELL_SIZE-4))
        
        # Draw ears
        ear_size = CELL_SIZE // 6
        pygame.draw.ellipse(mouse_img, MOUSE_EAR_COLOR, (4, 2, ear_size, ear_size))
        pygame.draw.ellipse(mouse_img, MOUSE_EAR_COLOR, (CELL_SIZE-ear_size-4, 2, ear_size, ear_size))
        
        # Draw eyes
        eye_size = CELL_SIZE // 10
        pygame.draw.ellipse(mouse_img, BLACK, (CELL_SIZE//3, CELL_SIZE//3, eye_size, eye_size))
        pygame.draw.ellipse(mouse_img, BLACK, (CELL_SIZE*2//3, CELL_SIZE//3, eye_size, eye_size))
        
        # Draw nose
        nose_size = CELL_SIZE // 10
        pygame.draw.ellipse(mouse_img, (255, 100, 100), (CELL_SIZE//2-nose_size//2, CELL_SIZE//2, nose_size, nose_size))
        
        # Draw tail
        tail_width = CELL_SIZE // 12
        tail_start = (CELL_SIZE//5, CELL_SIZE-tail_width)
        tail_end = (0, CELL_SIZE//2)
        pygame.draw.line(mouse_img, MOUSE_COLOR, tail_start, tail_end, tail_width)
        
        self.mouse_img = mouse_img
        
        # Create cheese image
        cheese_img = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        
        # Draw cheese base (a triangle)
        cheese_shape = [
            (CELL_SIZE//6, CELL_SIZE-CELL_SIZE//6),  # Bottom left
            (CELL_SIZE-CELL_SIZE//6, CELL_SIZE-CELL_SIZE//6),  # Bottom right
            (CELL_SIZE//2, CELL_SIZE//6)  # Top middle
        ]
        pygame.draw.polygon(cheese_img, CHEESE_COLOR, cheese_shape)
        
        # Add some holes to the cheese
        hole_positions = [
            (CELL_SIZE//3, CELL_SIZE//2), 
            (CELL_SIZE*2//3, CELL_SIZE*2//3),
            (CELL_SIZE//2, CELL_SIZE//3)
        ]
        for pos in hole_positions:
            hole_size = random.randint(CELL_SIZE//12, CELL_SIZE//8)
            pygame.draw.circle(cheese_img, BLACK, pos, hole_size)
        
        self.cheese_img = cheese_img
    
    def generate_maze(self):
        # Create a grid filled with walls
        width, height = GRID_WIDTH, GRID_HEIGHT
        
        # Initialize all cells as walls
        for y in range(height):
            for x in range(width):
                self.maze[y][x] = CellType.WALL
        
        # Create a grid for the maze generation algorithm
        # We need odd-indexed cells for passages and even-indexed cells for walls
        def carve_passage(x, y, visited):
            visited.add((x, y))
            # Set the current cell as a passage
            self.maze[y][x] = CellType.EMPTY
            
            # Define possible directions (up, right, down, left)
            directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
            random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # Check if the neighbor is valid and not visited
                if (0 <= nx < width and 0 <= ny < height and 
                    (nx, ny) not in visited and self.maze[ny][nx] == CellType.WALL):
                    # Carve a passage by making the wall between the cells empty
                    self.maze[y + dy//2][x + dx//2] = CellType.EMPTY
                    carve_passage(nx, ny, visited)
        
        # Start the maze generation from a random odd-indexed cell
        start_x = random.randrange(1, width, 2)
        start_y = random.randrange(1, height, 2)
        carve_passage(start_x, start_y, set())
        
        # Ensure start and goal positions are empty
        self.maze[self.start_pos[1]][self.start_pos[0]] = CellType.EMPTY
        self.maze[self.goal_pos[1]][self.goal_pos[0]] = CellType.EMPTY
        
        # Clear some random walls to create multiple paths
        for _ in range(width * height // 10):
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            if (x, y) != self.start_pos and (x, y) != self.goal_pos:
                self.maze[y][x] = CellType.EMPTY
        
        # Set start and goal positions
        self.maze[self.start_pos[1]][self.start_pos[0]] = CellType.START
        self.maze[self.goal_pos[1]][self.goal_pos[0]] = CellType.GOAL
        
        # Create a path between start and goal to ensure solvability
        self.ensure_path_exists()
        
        # Find path using A*
        self.find_path()
    
    def ensure_path_exists(self):
        # Simple method to ensure there's at least one valid path
        # by clearing walls in a direct path between start and goal
        sx, sy = self.start_pos
        gx, gy = self.goal_pos
        
        # Create a few random points to make the path less straight
        points = [(sx, sy)]
        for _ in range(3):
            px = random.randint(sx, gx) if sx < gx else random.randint(gx, sx)
            py = random.randint(sy, gy) if sy < gy else random.randint(gy, sy)
            points.append((px, py))
        points.append((gx, gy))
        
        # Clear walls along the path between consecutive points
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            # Create a straight line path between the points
            if x1 == x2:  # Vertical line
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    if (x1, y) != self.start_pos and (x1, y) != self.goal_pos:
                        self.maze[y][x1] = CellType.EMPTY
            elif y1 == y2:  # Horizontal line
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    if (x, y1) != self.start_pos and (x, y1) != self.goal_pos:
                        self.maze[y1][x] = CellType.EMPTY
            else:  # Diagonal line (approximated by a series of steps)
                dx = 1 if x2 > x1 else -1
                dy = 1 if y2 > y1 else -1
                x, y = x1, y1
                
                # Bresenham's line algorithm (simplified)
                if abs(x2 - x1) > abs(y2 - y1):
                    err = 0
                    derr = abs(y2 - y1) / abs(x2 - x1)
                    while x != x2:
                        if (x, y) != self.start_pos and (x, y) != self.goal_pos:
                            self.maze[y][x] = CellType.EMPTY
                        err += derr
                        if err >= 0.5:
                            y += dy
                            err -= 1
                        x += dx
                else:
                    err = 0
                    derr = abs(x2 - x1) / abs(y2 - y1)
                    while y != y2:
                        if (x, y) != self.start_pos and (x, y) != self.goal_pos:
                            self.maze[y][x] = CellType.EMPTY
                        err += derr
                        if err >= 0.5:
                            x += dx
                            err -= 1
                        y += dy
                
                if (x, y) != self.start_pos and (x, y) != self.goal_pos:
                    self.maze[y][x] = CellType.EMPTY
    
    def heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        
        # Check all four directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                # Check if not a wall
                if self.maze[ny][nx] != CellType.WALL:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def find_path(self):
        # Reset path variables
        self.path = []
        self.came_from = {}
        self.open_set = {self.start_pos}
        self.closed_set = set()
        self.g_score = {self.start_pos: 0}
        self.h_score = {self.start_pos: self.heuristic(self.start_pos, self.goal_pos)}  # Store h_score
        self.f_score = {self.start_pos: self.h_score[self.start_pos]}  # f = g + h, but g is 0 for start
        
        # Save search history for visualization
        self.search_path = []
        
        # A* algorithm
        open_heap = [(self.f_score[self.start_pos], 0, self.start_pos)]  # Add tiebreaker as second element
        heapq_counter = 1  # Counter for tiebreaking when f_scores are equal
        
        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            
            # Get neighbors for visualization
            neighbors = self.get_neighbors(current)
            
            # Skip if current is already in closed set
            if current in self.closed_set:
                continue
                
            # Save the search state BEFORE processing
            search_state = {
                'current': current,
                'open_set': self.open_set.copy(),
                'closed_set': self.closed_set.copy(),
                'g_score': self.g_score.copy(),
                'f_score': self.f_score.copy(),
                'h_score': self.h_score.copy(),  # Include h_score
                'neighbors': neighbors.copy()  # Save neighbors being considered
            }
            self.search_path.append(search_state)
            
            # Move current from open to closed
            if current in self.open_set:
                self.open_set.remove(current)
            self.closed_set.add(current)
            
            # Check if we've reached the goal
            if current == self.goal_pos:
                # The goal is now in the closed set
                
                # Capture final state for visualization
                search_state = {
                    'current': current,
                    'open_set': self.open_set.copy(),
                    'closed_set': self.closed_set.copy(),
                    'g_score': self.g_score.copy(),
                    'f_score': self.f_score.copy(),
                    'h_score': self.h_score.copy(),
                    'neighbors': []
                }
                self.search_path.append(search_state)
                
                # Reconstruct the path
                self.path = []
                while current in self.came_from:
                    self.path.append(current)
                    current = self.came_from[current]
                self.path.append(self.start_pos)
                self.path.reverse()
                return True
            
            # Process all neighbors
            for neighbor in neighbors:
                if neighbor in self.closed_set:
                    continue
                
                tentative_g_score = self.g_score.get(current, float('inf')) + 1
                
                if neighbor not in self.open_set or tentative_g_score < self.g_score.get(neighbor, float('inf')):
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    self.h_score[neighbor] = self.heuristic(neighbor, self.goal_pos)  # Store h_score
                    self.f_score[neighbor] = tentative_g_score + self.h_score[neighbor]
                    
                    if neighbor not in self.open_set:
                        self.open_set.add(neighbor)
                        heapq.heappush(open_heap, (self.f_score[neighbor], heapq_counter, neighbor))
                        heapq_counter += 1  # Increment counter to ensure unique second values
            
            # Capture state AFTER processing neighbors to show open nodes
            if current != self.goal_pos:  # Don't add duplicate state for goal
                search_state = {
                    'current': current,
                    'open_set': self.open_set.copy(),
                    'closed_set': self.closed_set.copy(),
                    'g_score': self.g_score.copy(),
                    'f_score': self.f_score.copy(),
                    'h_score': self.h_score.copy(),  # Include h_score
                    'neighbors': []  # No neighbors being examined at this point
                }
                self.search_path.append(search_state)
        
        return False
    
    def toggle_layer(self, layer_num):
        if layer_num == 1:
            self.show_maze = not self.show_maze
        elif layer_num == 2:
            self.show_graph = not self.show_graph
        elif layer_num == 3:
            self.show_path = not self.show_path
        elif layer_num == 4:
            self.show_scores = not self.show_scores
        elif layer_num == 5:
            self.show_f_score = not self.show_f_score
        elif layer_num == 6:
            self.show_g_score = not self.show_g_score
        elif layer_num == 7:
            self.show_h_score = not self.show_h_score
    
    def start_animation(self):
        if self.path:
            self.animation_active = True
            self.animation_step = 0
            self.animation_counter = 0
            self.mouse_pos = self.path[0]
    
    def start_search_animation(self):
        if self.search_path:
            self.search_animation_active = True
            self.search_animation_step = 0
            self.animation_counter = 0
            self.open_set = {self.start_pos}
            self.closed_set = set()
            self.current_node = None
            self.mouse_pos = self.start_pos
    
    def toggle_pause(self):
        """Toggle the pause state for animations"""
        self.paused = not self.paused
        return self.paused
    
    def update_animation(self):
        # If paused, don't update animation
        if self.paused:
            return
            
        # Update path following animation
        if self.animation_active and self.path:
            # Slow down the animation by using a counter
            self.animation_counter += 1
            if self.animation_counter >= PATH_ANIMATION_SPEED:
                self.animation_counter = 0
                if self.animation_step < len(self.path) - 1:
                    self.animation_step += 1
                    self.mouse_pos = self.path[self.animation_step]
                else:
                    self.animation_active = False
        
        # Update search animation
        if self.search_animation_active and self.search_path:
            # Check if we need to advance to the next search state
            self.animation_counter += 1
            if self.animation_counter >= self.search_animation_speed:  # Use adjustable speed
                self.animation_counter = 0
                
                if self.search_animation_step < len(self.search_path):
                    # Update state based on search history
                    state = self.search_path[self.search_animation_step]
                    self.open_set = state['open_set']
                    self.closed_set = state['closed_set']
                    self.current_node = state['current']
                    self.g_score = state['g_score']
                    self.f_score = state['f_score']
                    self.h_score = state['h_score']  # Update h_score
                    
                    # Move mouse to current position being evaluated
                    self.mouse_pos = self.current_node
                    
                    self.search_animation_step += 1
                else:
                    # End of search animation, start path animation
                    self.search_animation_active = False
                    self.current_node = None
                    self.start_animation()
    
    def adjust_search_speed(self, delta):
        """Adjust the search animation speed"""
        self.search_animation_speed = max(MIN_ANIMATION_SPEED, 
                                         min(MAX_ANIMATION_SPEED, 
                                             self.search_animation_speed + delta))
        return self.search_animation_speed

# Rendering functions
def draw_maze(screen, game_state):
    if game_state.show_maze:
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                if game_state.maze[y][x] == CellType.WALL:
                    # Create a 3D effect for walls
                    pygame.draw.rect(screen, DARK_BROWN, rect)
                    # Highlight top and left edges
                    pygame.draw.line(screen, BROWN, (x * CELL_SIZE, y * CELL_SIZE), 
                                    ((x+1) * CELL_SIZE, y * CELL_SIZE), 2)
                    pygame.draw.line(screen, BROWN, (x * CELL_SIZE, y * CELL_SIZE), 
                                    (x * CELL_SIZE, (y+1) * CELL_SIZE), 2)
                else:
                    pygame.draw.rect(screen, WHITE, rect)
                
                # Draw grid lines
                pygame.draw.rect(screen, BLACK, rect, 1)

def draw_graph(screen, game_state):
    if game_state.show_graph:
        # If in search animation, highlight all nodes with the same f-score
        if game_state.search_animation_active and game_state.current_node:
            current_f = game_state.f_score.get(game_state.current_node, float('inf'))
            
            # Find all nodes with the same f-score (which would be processed at similar priority)
            same_f_nodes = []
            for node in game_state.open_set:
                if game_state.f_score.get(node, float('inf')) == current_f:
                    same_f_nodes.append(node)
            
            # Draw a highlight for these nodes
            for x, y in same_f_nodes:
                center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
                radius = CELL_SIZE // 2 + 4  # Slightly larger than the node
                
                # Draw a glow effect
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, F_SCORE_INDICATOR_COLOR, (radius, radius), radius)
                screen.blit(s, (center[0]-radius, center[1]-radius))
        
        # Draw connections first so they're underneath the nodes
        for node, came_from in game_state.came_from.items():
            node_center = (node[0] * CELL_SIZE + CELL_SIZE // 2, node[1] * CELL_SIZE + CELL_SIZE // 2)
            came_from_center = (came_from[0] * CELL_SIZE + CELL_SIZE // 2, came_from[1] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.line(screen, BLUE, node_center, came_from_center, 2)
        
        # Draw closed nodes as circles
        for x, y in game_state.closed_set:
            # Skip if this is the current node - we'll draw it last
            if game_state.current_node and (x, y) == game_state.current_node:
                continue
                
            center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
            radius = CELL_SIZE // 2 - 2
            
            # Draw circle with a border
            pygame.draw.circle(screen, CLOSED_NODE_COLOR, center, radius)
            pygame.draw.circle(screen, BLACK, center, radius, 1)
            
            # Draw scores if enabled
            if game_state.show_scores:
                draw_cell_scores(screen, (x, y), game_state.f_score.get((x, y), float('inf')), 
                               game_state.g_score.get((x, y), float('inf')), game_state)
        
        # Draw open nodes as circles
        for x, y in game_state.open_set:
            # Skip if this is the current node - we'll draw it last
            if game_state.current_node and (x, y) == game_state.current_node:
                continue
                
            center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
            radius = CELL_SIZE // 2 - 2
            
            # Draw circle with a border
            pygame.draw.circle(screen, OPEN_NODE_COLOR, center, radius)
            pygame.draw.circle(screen, BLACK, center, radius, 1)
            
            # Draw scores if enabled
            if game_state.show_scores:
                draw_cell_scores(screen, (x, y), game_state.f_score.get((x, y), float('inf')), 
                               game_state.g_score.get((x, y), float('inf')), game_state)
        
        # Highlight current node being processed (draw last for visibility)
        if game_state.current_node:
            x, y = game_state.current_node
            center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
            radius = CELL_SIZE // 2 - 1  # Slightly larger radius
            
            # Draw a glowing effect
            glow_radius = radius + 4
            for r in range(glow_radius, radius-1, -1):
                alpha = 100 - (glow_radius - r) * 20
                s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
                s.fill((255, 140, 0, alpha))
                screen.blit(s, (center[0]-r, center[1]-r))
            
            # Draw highlighted current node
            pygame.draw.circle(screen, CURRENT_NODE_COLOR, center, radius)
            pygame.draw.circle(screen, BLACK, center, radius, 2)  # Thicker border
            
            # Draw scores if enabled
            if game_state.show_scores:
                draw_cell_scores(screen, (x, y), game_state.f_score.get((x, y), float('inf')), 
                               game_state.g_score.get((x, y), float('inf')), game_state)

def draw_cell_scores(screen, pos, f_score, g_score, game_state):
    """Draw f, g, and h scores for a cell with improved visibility and formatting"""
    x, y = pos
    cell_x = x * CELL_SIZE
    cell_y = y * CELL_SIZE
    
    if f_score != float('inf'):
        h_score = f_score - g_score  # h = f - g
        
        # Count how many score types are enabled
        enabled_score_count = 0
        if game_state.show_scores and game_state.show_f_score:
            enabled_score_count += 1
        if game_state.show_scores and game_state.show_g_score:
            enabled_score_count += 1
        if game_state.show_scores and game_state.show_h_score:
            enabled_score_count += 1
        
        # Choose font size based on how many scores are enabled
        if enabled_score_count == 1:
            score_font = FONT_SCORE_LARGE  # Largest font when only one score type is shown
            pill_padding = 4
        elif enabled_score_count == 2:
            score_font = FONT_SCORE_MEDIUM  # Medium font when two score types are shown
            pill_padding = 3
        else:
            score_font = FONT_SCORE  # Smallest font when all three are shown
            pill_padding = 2
        
        # Margin from cell edge - also adjust based on how many scores
        margin = 4 if enabled_score_count >= 3 else 6
        
        # Define positions for 1, 2, or 3 scores
        if enabled_score_count == 1:
            # If only one score is enabled, center it in the cell
            center_pos = (cell_x + CELL_SIZE//2, cell_y + CELL_SIZE//2)
            
            # Handle each possible single score
            if game_state.show_scores and game_state.show_f_score:
                draw_single_score(screen, center_pos, int(f_score), F_SCORE_COLOR, score_font, pill_padding)
            elif game_state.show_scores and game_state.show_g_score:
                draw_single_score(screen, center_pos, int(g_score), G_SCORE_COLOR, score_font, pill_padding)
            elif game_state.show_scores and game_state.show_h_score:
                draw_single_score(screen, center_pos, int(h_score), H_SCORE_COLOR, score_font, pill_padding)
        
        elif enabled_score_count == 2:
            # If two scores are enabled, position them on opposite corners
            top_pos = (cell_x + CELL_SIZE//2, cell_y + CELL_SIZE//3)
            bottom_pos = (cell_x + CELL_SIZE//2, cell_y + CELL_SIZE*2//3)
            
            # Draw the enabled scores in their positions
            if game_state.show_scores and game_state.show_f_score and game_state.show_g_score:
                draw_single_score(screen, top_pos, int(f_score), F_SCORE_COLOR, score_font, pill_padding)
                draw_single_score(screen, bottom_pos, int(g_score), G_SCORE_COLOR, score_font, pill_padding)
            elif game_state.show_scores and game_state.show_f_score and game_state.show_h_score:
                draw_single_score(screen, top_pos, int(f_score), F_SCORE_COLOR, score_font, pill_padding)
                draw_single_score(screen, bottom_pos, int(h_score), H_SCORE_COLOR, score_font, pill_padding)
            elif game_state.show_scores and game_state.show_g_score and game_state.show_h_score:
                draw_single_score(screen, top_pos, int(g_score), G_SCORE_COLOR, score_font, pill_padding)
                draw_single_score(screen, bottom_pos, int(h_score), H_SCORE_COLOR, score_font, pill_padding)
                
        else:
            # Original triangle formation for all three scores (or if nothing matches above)
            if game_state.show_scores and game_state.show_f_score:
                # F-score at top center
                f_text = score_font.render(f"{int(f_score)}", True, BLACK)
                f_rect = f_text.get_rect(center=(cell_x + CELL_SIZE//2, cell_y + margin + f_text.get_height()//2))
                
                # Draw background pill for f-score
                f_pill_rect = f_rect.inflate(pill_padding*2, pill_padding)
                f_pill_surface = pygame.Surface((f_pill_rect.width, f_pill_rect.height), pygame.SRCALPHA)
                f_pill_surface.fill((F_SCORE_COLOR[0], F_SCORE_COLOR[1], F_SCORE_COLOR[2], 220))  # Semi-transparent
                screen.blit(f_pill_surface, f_pill_rect)
                pygame.draw.rect(screen, BLACK, f_pill_rect, width=1)
                
                # Draw the score number
                screen.blit(f_text, f_rect)
            
            if game_state.show_scores and game_state.show_g_score:
                # G-score at bottom left
                g_text = score_font.render(f"{int(g_score)}", True, BLACK)
                g_rect = g_text.get_rect(midleft=(cell_x + margin, cell_y + CELL_SIZE - margin - g_text.get_height()//2))
                
                # Draw background pill for g-score
                g_pill_rect = g_rect.inflate(pill_padding*2, pill_padding)
                g_pill_surface = pygame.Surface((g_pill_rect.width, g_pill_rect.height), pygame.SRCALPHA)
                g_pill_surface.fill((G_SCORE_COLOR[0], G_SCORE_COLOR[1], G_SCORE_COLOR[2], 220))
                screen.blit(g_pill_surface, g_pill_rect)
                pygame.draw.rect(screen, BLACK, g_pill_rect, width=1)
                
                # Draw the score number
                screen.blit(g_text, g_rect)
            
            if game_state.show_scores and game_state.show_h_score:
                # H-score at bottom right
                h_text = score_font.render(f"{int(h_score)}", True, BLACK)
                h_rect = h_text.get_rect(midright=(cell_x + CELL_SIZE - margin, cell_y + CELL_SIZE - margin - h_text.get_height()//2))
                
                # Draw background pill for h-score
                h_pill_rect = h_rect.inflate(pill_padding*2, pill_padding)
                h_pill_surface = pygame.Surface((h_pill_rect.width, h_pill_rect.height), pygame.SRCALPHA)
                h_pill_surface.fill((H_SCORE_COLOR[0], H_SCORE_COLOR[1], H_SCORE_COLOR[2], 220))
                screen.blit(h_pill_surface, h_pill_rect)
                pygame.draw.rect(screen, BLACK, h_pill_rect, width=1)
                
                # Draw the score number
                screen.blit(h_text, h_rect)
        
        # Draw tiny labels in top-right corner only if any score is shown and we have all three scores
        if game_state.show_scores and enabled_score_count == 3:
            label_margin = 2
            label_size = 6
            label_font = pygame.font.SysFont("Arial", label_size)
            
            if game_state.show_f_score:
                # Draw F-label
                f_label = label_font.render("f", True, BLACK)
                f_label_rect = f_label.get_rect(topright=(cell_x + CELL_SIZE - label_margin, cell_y + label_margin))
                screen.blit(f_label, f_label_rect)
            
            if game_state.show_g_score:
                # Draw G-label
                g_label = label_font.render("g", True, BLACK)
                g_label_rect = g_label.get_rect(topright=(cell_x + CELL_SIZE - label_margin, 
                                              cell_y + label_margin + (label_size if game_state.show_f_score else 0)))
                screen.blit(g_label, g_label_rect)
            
            if game_state.show_h_score:
                # Draw H-label
                h_label = label_font.render("h", True, BLACK)
                offset = 0
                if game_state.show_f_score: offset += label_size
                if game_state.show_g_score: offset += label_size
                h_label_rect = h_label.get_rect(topright=(cell_x + CELL_SIZE - label_margin, cell_y + label_margin + offset))
                screen.blit(h_label, h_label_rect)

def draw_single_score(screen, pos, score, color, font, padding):
    """Helper function to draw a single score with its label"""
    # Draw the score with a pill background
    text = font.render(f"{score}", True, BLACK)
    text_rect = text.get_rect(center=pos)
    
    # Draw background pill
    pill_rect = text_rect.inflate(padding*4, padding*2)
    pill_surface = pygame.Surface((pill_rect.width, pill_rect.height), pygame.SRCALPHA)
    pill_surface.fill((color[0], color[1], color[2], 220))
    screen.blit(pill_surface, pill_rect)
    pygame.draw.rect(screen, BLACK, pill_rect, width=1)
    
    # Draw the text
    screen.blit(text, text_rect)

def draw_path(screen, game_state):
    if game_state.show_path and game_state.path:
        # Draw a thicker line for the path
        for i in range(len(game_state.path) - 1):
            start_pos = game_state.path[i]
            end_pos = game_state.path[i + 1]
            
            start_center = (start_pos[0] * CELL_SIZE + CELL_SIZE // 2, start_pos[1] * CELL_SIZE + CELL_SIZE // 2)
            end_center = (end_pos[0] * CELL_SIZE + CELL_SIZE // 2, end_pos[1] * CELL_SIZE + CELL_SIZE // 2)
            
            # Draw arrow-like path
            pygame.draw.line(screen, YELLOW, start_center, end_center, 4)
            
            # Draw a small circle at each node in the path
            pygame.draw.circle(screen, YELLOW, start_center, 5)
        
        # Draw the final point
        final_pos = game_state.path[-1]
        final_center = (final_pos[0] * CELL_SIZE + CELL_SIZE // 2, final_pos[1] * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(screen, YELLOW, final_center, 5)

def draw_mouse_and_cheese(screen, game_state):
    # Draw cheese at goal position
    cheese_rect = pygame.Rect(
        game_state.goal_pos[0] * CELL_SIZE, 
        game_state.goal_pos[1] * CELL_SIZE, 
        CELL_SIZE, CELL_SIZE
    )
    
    if game_state.cheese_img:
        screen.blit(game_state.cheese_img, cheese_rect)
    else:
        pygame.draw.rect(screen, CHEESE_COLOR, cheese_rect)
    
    # Draw mouse at current position
    mouse_rect = pygame.Rect(
        game_state.mouse_pos[0] * CELL_SIZE,
        game_state.mouse_pos[1] * CELL_SIZE,
        CELL_SIZE, CELL_SIZE
    )
    
    if game_state.mouse_img:
        # Determine rotation direction
        if game_state.animation_active and game_state.animation_step > 0:
            prev_pos = game_state.path[game_state.animation_step - 1]
            curr_pos = game_state.mouse_pos
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
        elif game_state.search_animation_active and game_state.search_animation_step > 1:
            # Get the previous position from the search path
            prev_state = game_state.search_path[game_state.search_animation_step - 2]
            prev_pos = prev_state['current']
            curr_pos = game_state.mouse_pos
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
        else:
            dx, dy = 0, 0
        
        # Rotate based on direction
        if dx > 0:  # Moving right
            rotated_mouse = pygame.transform.rotate(game_state.mouse_img, 270)
        elif dx < 0:  # Moving left
            rotated_mouse = pygame.transform.rotate(game_state.mouse_img, 90)
        elif dy > 0:  # Moving down
            rotated_mouse = pygame.transform.rotate(game_state.mouse_img, 180)
        else:  # Moving up or default
            rotated_mouse = game_state.mouse_img
        
        screen.blit(rotated_mouse, mouse_rect)
    else:
        pygame.draw.ellipse(screen, GRAY, mouse_rect)

def draw_sidebar(screen, game_state):
    # Draw the sidebar background
    sidebar_rect = pygame.Rect(MAZE_AREA_WIDTH, 0, SIDEBAR_WIDTH, HEIGHT)
    pygame.draw.rect(screen, PANEL_COLOR, sidebar_rect)
    pygame.draw.line(screen, BLACK, (MAZE_AREA_WIDTH, 0), (MAZE_AREA_WIDTH, HEIGHT), 2)
    
    # Draw title
    title = FONT_TITLE.render("A* Pathfinding", True, BLACK)
    screen.blit(title, (MAZE_AREA_WIDTH + 20, 20))
    
    subtitle = FONT_LARGE.render("Group 5 - Rat Maze", True, BLACK)
    screen.blit(subtitle, (MAZE_AREA_WIDTH + 20, 50))
    
    # Draw controls section
    control_y = 90
    controls_title = FONT_LARGE.render("Controls:", True, BLACK)
    screen.blit(controls_title, (MAZE_AREA_WIDTH + 20, control_y))
    
    # Draw button-like controls
    buttons = [
        {"key": "1", "action": "Toggle Maze", "active": game_state.show_maze},
        {"key": "2", "action": "Toggle Graph", "active": game_state.show_graph},
        {"key": "3", "action": "Toggle Path", "active": game_state.show_path},
        {"key": "4", "action": "Toggle All Scores", "active": game_state.show_scores},
        {"key": "5", "action": "Toggle F-Score", "active": game_state.show_f_score and game_state.show_scores},
        {"key": "6", "action": "Toggle G-Score", "active": game_state.show_g_score and game_state.show_scores},
        {"key": "7", "action": "Toggle H-Score", "active": game_state.show_h_score and game_state.show_scores},
        {"key": "spc", "action": "Start Path Animation", "active": game_state.animation_active},
        {"key": "A", "action": "Watch A* Search", "active": game_state.search_animation_active},
        {"key": "P", "action": "Pause Animation", "active": game_state.paused},
        {"key": "R", "action": "Generate New Maze", "active": False}
    ]
    
    button_width = SIDEBAR_WIDTH - 40
    button_height = 30
    button_margin = 8
    
    for i, button in enumerate(buttons):
        x = MAZE_AREA_WIDTH + 20
        y = control_y + 30 + i * (button_height + button_margin)
        
        # Draw button background
        button_color = LIGHT_GRAY
        if button["active"]:
            button_color = GREEN
        
        button_rect = pygame.Rect(x, y, button_width, button_height)
        pygame.draw.rect(screen, button_color, button_rect, border_radius=5)
        pygame.draw.rect(screen, BLACK, button_rect, width=1, border_radius=5)
        
        # Draw key in a circle
        key_circle_pos = (x + 15, y + button_height // 2)
        pygame.draw.circle(screen, WHITE, key_circle_pos, 10)
        pygame.draw.circle(screen, BLACK, key_circle_pos, 10, width=1)
        
        key_text = FONT_SMALL.render(button["key"], True, BLACK)
        key_rect = key_text.get_rect(center=key_circle_pos)
        screen.blit(key_text, key_rect)
        
        # Draw action text
        action_text = FONT_MEDIUM.render(button["action"], True, BLACK)
        screen.blit(action_text, (x + 30, y + button_height // 2 - action_text.get_height() // 2))
    
    # Draw animation speed controls
    speed_y = control_y + 30 + len(buttons) * (button_height + button_margin) + 10
    speed_title = FONT_MEDIUM.render("A* Search Speed:", True, BLACK)
    screen.blit(speed_title, (MAZE_AREA_WIDTH + 20, speed_y))
    
    # Display current speed
    speed_text = FONT_MEDIUM.render(f"Current: {game_state.search_animation_speed}", True, BLACK)
    screen.blit(speed_text, (MAZE_AREA_WIDTH + 150, speed_y))
    
    # Draw speed controls
    control_tips_y = speed_y + 25
    screen.blit(FONT_SMALL.render("Press + / - to adjust speed", True, BLACK), 
               (MAZE_AREA_WIDTH + 20, control_tips_y))
    screen.blit(FONT_SMALL.render("Lower = Slower", True, BLACK), 
               (MAZE_AREA_WIDTH + 20, control_tips_y + 20))
    
    # Show current animation status
    status_y = control_tips_y + 50
    if game_state.animation_active:
        status_text = FONT_MEDIUM.render(
            f"Path Animation: Step {game_state.animation_step} of {len(game_state.path)-1}", 
            True, BLACK
        )
        screen.blit(status_text, (MAZE_AREA_WIDTH + 20, status_y))
    elif game_state.search_animation_active:
        status_text = FONT_MEDIUM.render(
            f"Search Animation: Step {game_state.search_animation_step} of {len(game_state.search_path)}", 
            True, BLACK
        )
        screen.blit(status_text, (MAZE_AREA_WIDTH + 20, status_y))
        
        # Show current node info if available
        if game_state.current_node:
            current_info = FONT_MEDIUM.render(
                f"Evaluating node: {game_state.current_node}", 
                True, BLACK
            )
            screen.blit(current_info, (MAZE_AREA_WIDTH + 20, status_y + 20))
            
            # Show current f-score
            current_f = game_state.f_score.get(game_state.current_node, float('inf'))
            if current_f != float('inf'):
                f_score_info = FONT_MEDIUM.render(
                    f"Current f-score: {int(current_f)}", 
                    True, BLACK
                )
                screen.blit(f_score_info, (MAZE_AREA_WIDTH + 20, status_y + 40))

def draw_bottom_panel(screen, game_state):
    # Draw the bottom panel background
    bottom_rect = pygame.Rect(0, MAZE_AREA_HEIGHT, MAZE_AREA_WIDTH, BOTTOM_PANEL_HEIGHT)
    pygame.draw.rect(screen, PANEL_COLOR, bottom_rect)
    pygame.draw.line(screen, BLACK, (0, MAZE_AREA_HEIGHT), (MAZE_AREA_WIDTH, MAZE_AREA_HEIGHT), 2)
    
    # Define layout
    panel_padding = 20
    section_spacing = 40
    legend_width = MAZE_AREA_WIDTH * 0.6  # Legend takes 60% of the width
    
    # Draw legend section (left side)
    legend_x = panel_padding
    legend_y = MAZE_AREA_HEIGHT + panel_padding
    
    # Draw legend items - rearrange to put "Final Path" in the first column
    legend_items = [
        {"color": OPEN_NODE_COLOR, "label": "Open Nodes"},
        {"color": CLOSED_NODE_COLOR, "label": "Closed Nodes"},
        {"color": CURRENT_NODE_COLOR, "label": "Current Node"},
        {"color": YELLOW, "label": "Final Path"},
        {"color": F_SCORE_COLOR, "label": "F-Score (f = g + h)"},
        {"color": G_SCORE_COLOR, "label": "G-Score (start dist)"},
        {"color": H_SCORE_COLOR, "label": "H-Score (goal dist)"},
        {"color": F_SCORE_INDICATOR_COLOR, "label": "Same F-Score Group"}
    ]
    
    # Calculate how much space we need
    legend_start_y = legend_y + 30
    item_height = 25  # Reduced from 30 to fit better
    item_margin = 3   # Reduced from 5 to fit better
    items_per_column = 4
    
    # Calculate required height for legend box
    required_rows = math.ceil(len(legend_items) / 2)  # 2 columns
    required_height = required_rows * (item_height + item_margin) + 40  # 40 for padding and title
    
    # Draw legend title with a border box
    legend_title = FONT_LARGE.render("Legend:", True, BLACK)
    legend_title_rect = legend_title.get_rect(topleft=(legend_x, legend_y))
    legend_box = pygame.Rect(legend_x - 5, legend_y - 5, 
                           legend_width - panel_padding*2 + 10, 
                           required_height)
    pygame.draw.rect(screen, LIGHT_GRAY, legend_box, border_radius=5)
    pygame.draw.rect(screen, BLACK, legend_box, width=2, border_radius=5)
    screen.blit(legend_title, (legend_x, legend_y))
    
    # Organize legend in 2 columns
    column_width = legend_width / 2
    
    for i, item in enumerate(legend_items):
        col = i // items_per_column
        row = i % items_per_column
        
        x = legend_x + col * column_width
        y = legend_start_y + row * (item_height + item_margin)
        
        # Handle alpha value in color
        if len(item["color"]) > 3:
            # For colors with alpha, we need a special drawing approach
            sample_rect = pygame.Rect(x, y, 20, 20)
            # First draw underlying white to show transparency clearly
            pygame.draw.rect(screen, WHITE, sample_rect, border_radius=3)
            # Then draw our semi-transparent color on top
            s = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.rect(s, item["color"], (0, 0, 20, 20), border_radius=3)
            screen.blit(s, (x, y))
            pygame.draw.rect(screen, BLACK, sample_rect, width=1, border_radius=3)
        else:
            # For regular colors without alpha
            sample_rect = pygame.Rect(x, y, 20, 20)
            pygame.draw.rect(screen, item["color"], sample_rect, border_radius=3)
            pygame.draw.rect(screen, BLACK, sample_rect, width=1, border_radius=3)
        
        # Draw label
        label_text = FONT_MEDIUM.render(item["label"], True, BLACK)
        screen.blit(label_text, (x + 30, y + 10 - label_text.get_height() // 2))
    
    # Draw stats section (right side) - adjust vertical position to align with legend
    stats_x = legend_x + legend_width
    stats_y = MAZE_AREA_HEIGHT + panel_padding
    
    # Draw stats title with a border box
    stats_title = FONT_LARGE.render("Statistics:", True, BLACK)
    stats_title_rect = stats_title.get_rect(topleft=(stats_x, stats_y))
    stats_width = MAZE_AREA_WIDTH - legend_width - panel_padding * 2
    
    # Make stats box the same height as legend box for alignment
    stats_box = pygame.Rect(stats_x - 5, stats_y - 5, 
                          stats_width + panel_padding - 10, 
                          required_height)
    pygame.draw.rect(screen, LIGHT_GRAY, stats_box, border_radius=5)
    pygame.draw.rect(screen, BLACK, stats_box, width=2, border_radius=5)
    screen.blit(stats_title, (stats_x, stats_y))
    
    # Show path length
    path_length = len(game_state.path) - 1 if game_state.path else 0
    path_text = FONT_MEDIUM.render(f"Path Length: {path_length} steps", True, BLACK)
    screen.blit(path_text, (stats_x + 10, stats_y + 40))
    
    # Show nodes explored
    nodes_explored = len(game_state.closed_set)
    nodes_text = FONT_MEDIUM.render(f"Nodes Explored: {nodes_explored}", True, BLACK)
    screen.blit(nodes_text, (stats_x + 10, stats_y + 70))
    
    # Add an explanation of the A* animation
    if game_state.search_animation_active:
        explanation_text = FONT_SMALL.render("A* explores nodes in order of lowest f-score first", True, BLACK)
        screen.blit(explanation_text, (stats_x + 10, stats_y + 100))
        explanation_text2 = FONT_SMALL.render("Highlighted nodes have the same f-score priority", True, BLACK)
        screen.blit(explanation_text2, (stats_x + 10, stats_y + 120))

def main():
    # Set up the display
    screen = pygame.display.set_mode((MAZE_AREA_WIDTH + SIDEBAR_WIDTH, HEIGHT))
    pygame.display.set_caption("A* Pathfinding Rat Maze")
    
    # Set up the clock
    clock = pygame.time.Clock()
    
    # Create the game state
    game_state = GameState()
    
    # Main game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    game_state.toggle_layer(1)
                elif event.key == pygame.K_2:
                    game_state.toggle_layer(2)
                elif event.key == pygame.K_3:
                    game_state.toggle_layer(3)
                elif event.key == pygame.K_4:
                    game_state.toggle_layer(4)
                elif event.key == pygame.K_5:
                    game_state.toggle_layer(5)
                elif event.key == pygame.K_6:
                    game_state.toggle_layer(6)
                elif event.key == pygame.K_7:
                    game_state.toggle_layer(7)
                elif event.key == pygame.K_SPACE:
                    game_state.start_animation()
                elif event.key == pygame.K_a:
                    # Start A* search animation
                    game_state.start_search_animation()
                elif event.key == pygame.K_p:
                    # Toggle pause/resume of animation
                    game_state.toggle_pause()
                elif event.key == pygame.K_r:
                    game_state = GameState()  # Generate new maze
                # Speed control keys
                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                    # Decrease speed value (makes animation slower)
                    game_state.adjust_search_speed(-1)
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    # Increase speed value (makes animation faster)
                    game_state.adjust_search_speed(1)
        
        # Update game state
        if game_state.animation_active or game_state.search_animation_active:
            game_state.update_animation()
        
        # Clear the screen
        screen.fill(WHITE)
        
        # Draw maze area
        draw_maze(screen, game_state)
        draw_graph(screen, game_state)
        draw_path(screen, game_state)
        draw_mouse_and_cheese(screen, game_state)
        
        # Draw bottom panel (legend and statistics)
        draw_bottom_panel(screen, game_state)
        
        # Draw sidebar
        draw_sidebar(screen, game_state)
        
        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 