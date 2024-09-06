from pyrr import Vector3, vector, vector3, matrix44
from math import sin, cos, radians

class Camera:
    def __init__(self):
        self.radius = 5.0  # Distance from the target (zoom level)
        self.camera_pos = Vector3([0.0, 0.0, self.radius])
        self.target = Vector3([0.0, 0.0, 0.0])  # The point around which the camera orbits
        self.camera_up = Vector3([0.0, 1.0, 0.0])

        self.mouse_sensitivity = 0.5
        self.zoom_sensitivity = 1.0  # Adjusted for smoother zooming
        self.jaw = 90.0  # Initial yaw (horizontal angle)
        self.pitch = 0.0  # Initial pitch (vertical angle)

        self.update_camera_vectors()

    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity

        self.jaw += xoffset
        self.pitch += yoffset

        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0

        self.update_camera_vectors()

    def process_mouse_scroll(self, yoffset):
        self.radius -= yoffset * self.zoom_sensitivity
        if self.radius < 1.0:
            self.radius = 1.0  # Prevent the camera from getting too close to the target
        elif self.radius > 50.0:  # Adding an upper limit to prevent zooming out too far
            self.radius = 50.0

        self.update_camera_vectors()

    def update_camera_vectors(self):
        # Calculate the new camera position based on the spherical coordinates
        x = self.radius * cos(radians(self.jaw)) * cos(radians(self.pitch))
        y = self.radius * sin(radians(self.pitch))
        z = self.radius * sin(radians(self.jaw)) * cos(radians(self.pitch))

        self.camera_pos = Vector3([x, y, z]) + self.target

        # The front vector is the direction from the camera position to the target
        self.camera_front = vector.normalise(self.target - self.camera_pos)

        # Recalculate the right and up vectors
        self.camera_right = vector.normalise(vector3.cross(self.camera_front, Vector3([0.0, 1.0, 0.0])))
        self.camera_up = vector.normalise(vector3.cross(self.camera_right, self.camera_front))

    def get_view_matrix(self):
        return matrix44.create_look_at(self.camera_pos, self.target, self.camera_up)
