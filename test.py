import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from ObjLoader import ObjLoader
from pyrr import Vector3, matrix44
from tkinter import Tk, filedialog
from camera import Camera
from TextureLoader import load_texture
import numpy as np
import matplotlib.path as mplPath
import pandas as pd  # Import pandas for Excel export


# Initialize the main camera
cam = Camera()
WIDTH, HEIGHT = 1280, 720
lastX, lastY = WIDTH / 2, HEIGHT / 2
first_mouse = True
left_mouse_button_pressed = False

# Global variables for selection
selection_mode = False
polygon_vertices = []
selected_triangles = set()

# Shader source code for the model
vertex_src = """
# version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texcoord;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

out vec2 v_texcoord;

void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_texcoord = a_texcoord;
}
"""

fragment_src = """
# version 330 core
in vec2 v_texcoord;
out vec4 out_color;

uniform sampler2D s_texture;

void main()
{
    out_color = texture(s_texture, v_texcoord);
}
"""

# Shader source code for 2D line rendering
line_vertex_src = """
# version 330 core
layout(location = 5) in vec2 a_position;

void main()
{
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

line_fragment_src = """
# version 330 core
out vec4 out_color;

void main()
{
    out_color = vec4(1.0, 1.0, 1.0, 0.5);  // Yellow color for the selection polygon
}
"""

# Function to load the object file
def load_object_file():
    Tk().withdraw()  # Hide the main tkinter window
    filepath = filedialog.askopenfilename(filetypes=[("Object Files", "*.obj")])
    if filepath:
        return filepath
    return None

""" def save_vertex_coordinates_to_excel(object_buffer, filename="vertex_coordinates.xlsx"):
    try:
        # Each vertex has 8 components: 3 for position, 3 for normals, and 2 for texture coordinates
        num_vertices = len(object_buffer) // 8
        
        # Extract the vertex positions
        vertices = []
        for i in range(num_vertices):
            x, y, z = object_buffer[i * 8], object_buffer[i * 8 + 1], object_buffer[i * 8 + 2]
            vertices.append([x, y, z])
        
        # Create a DataFrame with the vertex data
        df = pd.DataFrame(vertices, columns=["X", "Y", "Z"])
        
        # Save the DataFrame to an Excel file
        df.to_excel(filename, index=False)
        print(f"Vertex coordinates saved to {filename}")
    except Exception as e:
        print(f"Failed to save vertex coordinates: {e}") """

# Function to unproject screen coordinates to world coordinates
def get_world_coordinates_from_click(window, xpos, ypos):
    # Convert window coordinates to NDC
    ndc_x = (2.0 * xpos) / WIDTH - 1.0
    ndc_y = 1.0 - (2.0 * ypos) / HEIGHT  # Flip Y-axis
    ndc_z = 1.0  # Assuming near plane, adjust if you need to calculate for depth
    ndc_coords = np.array([ndc_x, ndc_y, ndc_z, 1.0])
    #print(ndc_coords)

    # Invert the projection and view matrices
    inv_projection = np.linalg.inv(projection)
    #print(inv_projection)
    inv_view = np.linalg.inv(cam.get_view_matrix())

    # Convert NDC to world coordinates
    world_coords = inv_view @ inv_projection @ ndc_coords
    world_coords = world_coords / world_coords[3]  # Normalize by w to get world coordinates

    return world_coords[:3]  # Return as (x, y, z) in world space

# Mouse button callback for selection and model movement
def mouse_button_clb(window, button, action, mods):
    global polygon_vertices, selection_mode, left_mouse_button_pressed

    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            xpos, ypos = glfw.get_cursor_pos(window)
            world_coords = get_world_coordinates_from_click(window, xpos, ypos)

            if selection_mode:
                # Add the current mouse position to the polygon vertices
                polygon_vertices.append((xpos, ypos))
                print(f"Vertex added to polygon: ({xpos}, {ypos})")
            else:
                left_mouse_button_pressed = True
        elif action == glfw.RELEASE:
            left_mouse_button_pressed = False

# Keyboard input callback to toggle selection mode and delete triangles
def key_input_clb(window, key, scancode, action, mode):
    global selection_mode, polygon_vertices, selected_triangles

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if key == glfw.KEY_D:

        if action == glfw.PRESS:
            
            selection_mode = True
            polygon_vertices = []
            selected_triangles = set()
            print("Selection mode activated.")
        elif action == glfw.RELEASE:
            selection_mode = False
            # Perform selection
            perform_selection(polygon_vertices)
            print("Selection mode deactivated.")
        

    # Press 'X' to delete selected triangles
    if key == glfw.KEY_X and action == glfw.PRESS:
        delete_selected_triangles()
        print("Selected triangles deleted.")

# Mouse movement callback for camera rotation
def mouse_look_clb(window, xpos, ypos):
    global first_mouse, lastX, lastY, left_mouse_button_pressed

    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False

    if left_mouse_button_pressed:  # Rotate only when left mouse button is pressed
        xoffset = xpos - lastX
        yoffset = lastY - ypos

        cam.process_mouse_movement(xoffset, yoffset)

    lastX = xpos
    lastY = ypos

# Scroll callback for zooming
def scroll_clb(window, xoffset, yoffset):
    cam.process_mouse_scroll(yoffset)

def perform_selection(polygon_vertices):
    global object_indices, object_buffer, selected_triangles

    if len(polygon_vertices) < 3:
        print("Polygon is not valid (less than 3 vertices). No selection made.")
        return

    # Convert screen coordinates to OpenGL normalized device coordinates
    # This function is used to check whether 
    polygon_path = mplPath.Path(polygon_vertices)

    for i in range(len(object_indices) // 3): # Search trough all the primitives
        triangle_vertices = []
        for j in range(3):
            vertex_index = object_indices[i * 3 + j] #Extract indexes of vertices
            vertex_position = object_buffer[(i * 3 + j) * 8: (i * 3 + j) * 8 + 3]  # Extract the vertex position.. object_buffer format 3 vertices, 2 texture, 3 normals
            screen_position = project_to_screen(vertex_position)  #Projecting vertex positions to screen coordinates
            triangle_vertices.append(screen_position)
            print("\nObject indices..... ", object_indices[i * 3 + j]," \nVertex......\n", vertex_position)

        # Check if any vertex of the triangle is within the polygon
        if any(polygon_path.contains_point(vertex) for vertex in triangle_vertices):
            selected_triangles.add(i)
            print(f"Triangle {i} selected.")

def project_to_screen(vertex_position):
    # Convert the vertex position to 4D homogeneous coordinates
    vertex_position_4d = np.array([vertex_position[0], vertex_position[1], vertex_position[2], 1.0])

    # Apply the model, view, and projection matrices
    transformed_vertex1 = model_position @ vertex_position_4d
    transformed_vertex2 = cam.get_view_matrix() @ transformed_vertex1
    transformed_vertex3 = projection @ transformed_vertex2
    # Perform the perspective divide to transform to normalized device coordinates (NDC)
    ndc_coords = vertex_position_4d[:3] / vertex_position_4d[2]

    # Convert NDC to screen space coordinates
    screen_x = vertex_position_4d[0] * WIDTH / 2 + WIDTH / 2 + 500
    screen_y =  vertex_position_4d[1] * HEIGHT / 2 + HEIGHT / 2 + 200

    print("-------------------------------------------------------")
    print("\nModel Position Matrix ......\n",model_position)
    print("\nModel Transform:",transformed_vertex1)
    print("\nView Matrix ......\n",cam.get_view_matrix())
    print("\nView Transform:",transformed_vertex2)
    print("\nProjection Matrix...........\n",projection) 
    print("\nProjection Transform:",transformed_vertex3)
    print("\nNDC coords........",ndc_coords)
    print("\nScreen coordinates.........", screen_x,screen_y)
    print("--------------------------------------------------------")

    return screen_x, screen_y

# Render the selection polygon
def render_selection_polygon(vertices):
    if len(vertices) < 2:
        return  # Need at least 2 points to draw a line

    glUseProgram(line_shader)
    
    # Convert the polygon vertices to NDC coordinates
    polygon_ndc = []
    for vertex in vertices:
        ndc_x = (vertex[0] / WIDTH) * 2.0 - 1.0
        ndc_y = 1.0 - (vertex[1] / HEIGHT) * 2.0
        #print(vertex)
        polygon_ndc.append([ndc_x, ndc_y])

    polygon_ndc = np.array(polygon_ndc, dtype=np.float32)

    # Generate a VBO for the line vertices
    line_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, line_vbo)
    glBufferData(GL_ARRAY_BUFFER, polygon_ndc.nbytes, polygon_ndc, GL_STATIC_DRAW)

    glEnableVertexAttribArray(5)
    glVertexAttribPointer(5, 2, GL_FLOAT, GL_FALSE, 0, None)

    # Draw the lines connecting the vertices
    glDrawArrays(GL_LINE_LOOP, 0, len(polygon_ndc))

    glDisableVertexAttribArray(5)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glDeleteBuffers(1, [line_vbo])

    glUseProgram(0)

def delete_selected_triangles():
    global object_indices, object_buffer, selected_triangles, VBO, VAO

    #print("object buffer:", object_buffer)
    #print("object indices:", object_indices)

    # Convert to lists for easier manipulation
    object_buffer_list = object_buffer.tolist()
    object_indices_list = object_indices.tolist()

    # Create a set of vertices to delete based on the selected triangles
    vertices_to_delete = []
    selected_triangles = sorted(selected_triangles, reverse=True)
    print("\n selected triangles list:", selected_triangles)
    for triangle_index in selected_triangles:
        start_index = triangle_index * 3
        del object_indices_list[start_index : start_index + 3]
  
    # Filter out the vertices and indices that need to be deleted
    #new_object_buffer_list = [v for i, v in enumerate(object_buffer_list) if i // 8 not in vertices_to_delete]
    #new_object_indices_list = [i for i in object_indices_list if i not in vertices_to_delete]

    # Convert back to numpy arrays
    object_buffer = np.array(object_buffer_list, dtype=np.float32)
    object_indices = np.array(object_indices_list, dtype=np.uint32)
    print("\nnakon brisanja indices", object_indices)
    print("\nnakon brisanja", object_buffer)
   
    # Re-upload the updated buffer and indices to the GPU
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, object_buffer.nbytes, object_buffer, GL_DYNAMIC_DRAW)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # Re-enable the vertex attributes
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, object_buffer.itemsize * 8, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, object_buffer.itemsize * 8, ctypes.c_void_p(object_buffer.itemsize * 3))

    # Clear the selected triangles
    selected_triangles.clear()


# Window resize callback
def window_resize_clb(window, width, height):
    global WIDTH, HEIGHT, projection
    WIDTH, HEIGHT = width, height
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(60, width / height, 0.1, 100)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW cannot be initialized!")

# Load the 3D model file
obj_file_path = load_object_file()
if not obj_file_path:
    raise Exception("No object file selected!")

# Create the window
window = glfw.create_window(WIDTH, HEIGHT, "My OpenGL window", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window cannot be created!")

glfw.set_window_pos(window, 400, 200)

# Set callbacks
glfw.set_window_size_callback(window, window_resize_clb)
glfw.set_cursor_pos_callback(window, mouse_look_clb)
glfw.set_mouse_button_callback(window, mouse_button_clb)
glfw.set_scroll_callback(window, scroll_clb)
glfw.set_key_callback(window, key_input_clb)

# Display the mouse cursor
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

# Make the context current
glfw.make_context_current(window)

# Load the selected 3D model
try:
    object_indices, object_buffer, texture_path = ObjLoader.load_model(obj_file_path)
    print("Model loaded successfully.")

    #save_vertex_coordinates_to_excel(object_buffer)  # Save vertex coordinates to Excel after loading the model
except FileNotFoundError as e:
    glfw.terminate()
    raise Exception(f"Failed to load model: {e}")

""" print("\object buffer iza poziva:", object_buffer) """

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
line_shader = compileProgram(compileShader(line_vertex_src, GL_VERTEX_SHADER), compileShader(line_fragment_src, GL_FRAGMENT_SHADER))

# Generate and load the texture
if texture_path:
    texture = glGenTextures(1)
    load_texture(texture_path, texture)

# Create VAO and VBO for the model
VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)

glBindVertexArray(VAO)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, object_buffer.nbytes, object_buffer, GL_DYNAMIC_DRAW)

""" EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, object_indices.nbytes, object_indices, GL_DYNAMIC_DRAW) """

# Enable position attribute
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, object_buffer.itemsize * 8, ctypes.c_void_p(0))

# Enable texture coordinate attribute
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, object_buffer.itemsize * 8, ctypes.c_void_p(object_buffer.itemsize * 3))

# Set the polygon mode to fill to render with texture
glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
glUseProgram(shader)
glClearColor(0, 0, 0, 1)
glEnable(GL_DEPTH_TEST)

projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, 30)
#print(projection)
model_position = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]))
#print(model_position)

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")

glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

# Bind the texture
glActiveTexture(GL_TEXTURE0)
glBindTexture(GL_TEXTURE_2D, texture)
glUniform1i(glGetUniformLocation(shader, "s_texture"), 0)

def print_screen_coordinates(object_buffer, model_position, view_matrix, projection_matrix):
    num_vertices = len(object_buffer) // 8  # Assuming each vertex has 8 attributes: 3 position, 2 texture, 3 normal
    
    for i in range(num_vertices):
        # Extract the vertex position (x, y, z)
        vertex_position = np.array([object_buffer[i * 8], object_buffer[i * 8 + 1], object_buffer[i * 8 + 2], 1.0])
        
        # Apply model, view, and projection transformations
        transformed_vertex = projection_matrix @ view_matrix @ model_position @ vertex_position
        
        # Perform perspective division to get NDC (Normalized Device Coordinates)
        ndc_coords = transformed_vertex[:3] / transformed_vertex[3]  # Divide by w (homogeneous coordinate)
        
        # Convert NDC to screen coordinates
        screen_x = int((ndc_coords[0] + 1) * WIDTH / 2)
        screen_y = int((1 - ndc_coords[1]) * HEIGHT / 2)  # Y axis is flipped in screen space
        
        print(f"Vertex {i}: Screen coordinates -> X: {screen_x}, Y: {screen_y}")

# Example usage in your main loop or after loading the model:
#print_screen_coordinates(object_buffer, model_position, cam.get_view_matrix(), projection)

# Set the polygon mode to wireframe (line) to render the model in wireframe mode
glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

# Main application loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Render the main model in wireframe
    glUseProgram(shader)
    view = cam.get_view_matrix()
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glBindVertexArray(VAO)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_position)

    # Drawing loop in wireframe
    glDrawArrays(GL_TRIANGLES, 0, len(object_indices))

    # Render the selection polygon (optional)
    if selection_mode:
        render_selection_polygon(polygon_vertices)

    glfw.swap_buffers(window)

# Cleanup resources
glDeleteVertexArrays(1, [VAO])
glDeleteBuffers(1, [VBO])
glDeleteTextures([texture])

glfw.terminate()