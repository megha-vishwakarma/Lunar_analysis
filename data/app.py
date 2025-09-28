import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import heapq
from scipy.spatial import Delaunay
import trimesh
import trimesh.visual

ALGORITHM_ASTAR = "A*"
ALGORITHM_DIJKSTRA = "Dijkstra"
ALGORITHM_BFS = "BFS"
ALGORITHM_DFS = "DFS"

class PathFinder:
    def __init__(self, image):
        self.image = image
        self.safe_zones = self.identify_safe_zones()

    def identify_safe_zones(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        # Considering white and grey as safe zones and black as unsafe
        _, safe_zones = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        return safe_zones

    def heuristic(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def find_path_astar(self, start, end):
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, end)}
        oheap = []

        heapq.heappush(oheap, (fscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == end:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data.append(start)
                data.reverse()
                return data

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + self.heuristic(current, neighbor)

                if 0 <= neighbor[0] < self.safe_zones.shape[1] and 0 <= neighbor[1] < self.safe_zones.shape[0]:
                    if self.safe_zones[neighbor[1], neighbor[0]] == 0:
                        continue
                else:
                    continue

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue

                if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, end)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return []

    def find_path_dijkstra(self, start, end):
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        oheap = []

        heapq.heappush(oheap, (gscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == end:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data.append(start)
                data.reverse()
                return data

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + self.heuristic(current, neighbor)

                if 0 <= neighbor[0] < self.safe_zones.shape[1] and 0 <= neighbor[1] < self.safe_zones.shape[0]:
                    if self.safe_zones[neighbor[1], neighbor[0]] == 0:
                        continue
                else:
                    continue

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue

                if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    heapq.heappush(oheap, (gscore[neighbor], neighbor))

        return []

    def find_path_bfs(self, start, end):
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        queue = [start]

        while queue:
            current = queue.pop(0)

            if current == end:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data.append(start)
                data.reverse()
                return data

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j

                if 0 <= neighbor[0] < self.safe_zones.shape[1] and 0 <= neighbor[1] < self.safe_zones.shape[0]:
                    if self.safe_zones[neighbor[1], neighbor[0]] == 0:
                        continue
                else:
                    continue

                if neighbor in close_set:
                    continue

                if neighbor not in queue:
                    came_from[neighbor] = current
                    queue.append(neighbor)

        return []

    def find_path_dfs(self, start, end):
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        stack = [start]

        while stack:
            current = stack.pop()

            if current == end:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data.append(start)
                data.reverse()
                return data

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j

                if 0 <= neighbor[0] < self.safe_zones.shape[1] and 0 <= neighbor[1] < self.safe_zones.shape[0]:
                    if self.safe_zones[neighbor[1], neighbor[0]] == 0:
                        continue
                else:
                    continue

                if neighbor in close_set:
                    continue

                if neighbor not in stack:
                    came_from[neighbor] = current
                    stack.append(neighbor)

        return []

class LunarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lunar Path Finder")

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.process_button = tk.Button(root, text="Process Image", command=self.process_image)
        self.process_button.pack()

        self.start_button = tk.Button(root, text="Set Start Point", command=self.enable_start_point_selection)
        self.start_button.pack()

        self.end_button = tk.Button(root, text="Set End Point", command=self.enable_end_point_selection)
        self.end_button.pack()

        self.find_path_button = tk.Button(root, text="Find Path", command=self.find_path)
        self.find_path_button.pack()

        self.save_button = tk.Button(root, text="Save Path Image", command=self.save_path_image)
        self.save_button.pack()

        self.convert_button = tk.Button(root, text="Convert to 3D Model", command=self.convert_to_3d_model)
        self.convert_button.pack()
        
         # Create scrollbars
        self.scrollbarx = tk.Scrollbar(root, orient=tk.HORIZONTAL)
        self.scrollbary = tk.Scrollbar(root, orient=tk.VERTICAL)
        self.scrollbarx.pack(side=tk.BOTTOM, fill=tk.X)
        self.scrollbary.pack(side=tk.RIGHT, fill=tk.Y)
        
       


        self.algorithm_var = tk.StringVar(value=ALGORITHM_ASTAR)
        self.algorithm_menu = tk.OptionMenu(root, self.algorithm_var, ALGORITHM_ASTAR, ALGORITHM_DIJKSTRA, ALGORITHM_BFS, ALGORITHM_DFS)
        self.algorithm_menu.pack()

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()
        # Configure canvas to use scrollbars
        self.canvas.configure(xscrollcommand=self.scrollbarx.set, yscrollcommand=self.scrollbary.set)
        self.scrollbarx.config(command=self.canvas.xview)
        self.scrollbary.config(command=self.canvas.yview)
        
        self.image = None
        self.image_array = None
        self.start_point = None
        self.end_point = None
        self.path_finder = None
        self.path = None

        self.selecting_start_point = False
        self.selecting_end_point = False

        self.canvas.bind("<Button-1>", self.canvas_click)

        # Zoom functionality
        self.scale_factor = 1.0
        self.canvas.bind("<MouseWheel>", self.zoom)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            self.image_array = np.array(self.image)
            self.display_image(self.image)

    def process_image(self):
        if self.image_array is not None:
            self.path_finder = PathFinder(self.image_array)
            safe_zone_image = cv2.cvtColor(self.path_finder.safe_zones, cv2.COLOR_GRAY2RGB)
            self.display_image(Image.fromarray(safe_zone_image))

    def enable_start_point_selection(self):
        self.selecting_start_point = True
        self.selecting_end_point = False

    def enable_end_point_selection(self):
        self.selecting_start_point = False
        self.selecting_end_point = True

    def find_path(self):
        if self.path_finder and self.start_point and self.end_point:
            algorithm = self.algorithm_var.get()
            if algorithm == ALGORITHM_ASTAR:
                self.path = self.path_finder.find_path_astar(self.start_point, self.end_point)
            elif algorithm == ALGORITHM_DIJKSTRA:
                self.path = self.path_finder.find_path_dijkstra(self.start_point, self.end_point)
            elif algorithm == ALGORITHM_BFS:
                self.path = self.path_finder.find_path_bfs(self.start_point, self.end_point)
            elif algorithm == ALGORITHM_DFS:
                self.path = self.path_finder.find_path_dfs(self.start_point, self.end_point)

            self.display_image(self.image)  # Reset to the original image
            self.draw_path()

    def save_path_image(self):
        if self.image and self.path:
            path_image = self.image.copy()
            draw = ImageDraw.Draw(path_image)
            for i in range(len(self.path) - 1):
                draw.line(
                    (
                        self.path[i][0], self.path[i][1],
                        self.path[i + 1][0], self.path[i + 1][1]
                    ),
                    fill="red", width=2
                )
            file_path = filedialog.asksaveasfilename(defaultextension=".tiff", filetypes=[("TIFF files", "*.tiff")])
            if file_path:
                path_image.save(file_path, format="TIFF")

    def convert_to_3d_model(self):
        if self.image_array is not None:
            # Convert the image to grayscale and get dimensions
            height_map = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
            x = np.arange(height_map.shape[1])
            y = np.arange(height_map.shape[0])
            x, y = np.meshgrid(x, y)
            z = height_map.astype(np.float32)
    
            # Normalize the height values to the range 0-1
            z = (z - np.min(z)) / (np.max(z) - np.min(z))
    
            # Scale the height values to a smaller range to make the model smoother
            z *= 8  # Adjust this factor as needed for desired smoothness
    
            # Adding a small Gaussian filter to smooth the surface
            z_smooth = cv2.GaussianBlur(z, (5, 5), 0)
    
            # Creating the mesh using the smoothed height map
            points = np.c_[x.flatten(), y.flatten(), z_smooth.flatten()]
            tri = Delaunay(points[:, :2])
            mesh = trimesh.Trimesh(vertices=points, faces=tri.simplices)
    
            # Texture mapping the original image onto the 3D model
            texture = trimesh.visual.TextureVisuals(image=self.image_array)
            mesh.visual = texture
    
            if self.path:
                path_points = np.array(self.path)
                path_z = z[path_points[:, 1], path_points[:, 0]]
                path_points_3d = np.c_[path_points, path_z]
                path_mesh = trimesh.load_path(path_points_3d)
                mesh = mesh + path_mesh
    
            # Export to GLTF
            file_path = filedialog.asksaveasfilename(defaultextension=".glb", filetypes=[("GLTF files", "*.glb"), ("GLTF files", "*.gltf")])
            if file_path:
                mesh.export(file_path)
                messagebox.showinfo("3D Model Saved", f"3D model saved as {file_path}")
        else:
            messagebox.showerror("Error", "No image loaded.")

   
    def canvas_click(self, event):           
        if self.image:
           # Adjust coordinates for canvas scroll position
           x = int(self.canvas.canvasx(event.x) / self.scale_factor)
           y = int(self.canvas.canvasy(event.y) / self.scale_factor)
           
           if self.selecting_start_point:
               self.start_point = (x, y)
               self.draw_point(x, y, "green")
               self.selecting_start_point = False
           elif self.selecting_end_point:
               self.end_point = (x, y)
               self.draw_point(x, y, "red")
               self.selecting_end_point = False

    def display_image(self, image):
        self.canvas.delete("all")  # Clear the canvas
        self.canvas.image = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas.image)
        if self.path:
            self.draw_path()
        # Configure scroll region
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def draw_point(self, x, y, color):
      
        self.canvas.create_oval(
            x * self.scale_factor - 3, y * self.scale_factor - 3,
            x * self.scale_factor + 3, y * self.scale_factor + 3,
            fill=color, outline=color
        )

    def draw_path(self):
        if self.path:
            for i in range(len(self.path) - 1):
                self.canvas.create_line(
                    self.path[i][0] * self.scale_factor, self.path[i][1] * self.scale_factor,
                    self.path[i + 1][0] * self.scale_factor, self.path[i + 1][1] * self.scale_factor,
                    fill="red", width=2
                )

    def zoom(self, event):
        if event.delta > 0:
            self.scale_factor *= 1.1
        elif event.delta < 0:
            self.scale_factor /= 1.1

        width, height = int(self.image.width * self.scale_factor), int(self.image.height * self.scale_factor)
        resized_image = self.image.resize((width, height), Image.LANCZOS)
        self.display_image(resized_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = LunarApp(root)
    root.mainloop()
