import numpy as np
import os

class ObjLoader:
    buffer = []

    @staticmethod
    def search_data(data_values, coordinates, skip, data_type):
        for d in data_values:
            if d == skip:
                continue
            if data_type == 'float':
                coordinates.append(float(d))
            elif data_type == 'int':
                coordinates.append(int(d) - 1)

    @staticmethod
    def create_sorted_vertex_buffer(indices_data, vertices, textures, normals):
        for i, ind in enumerate(indices_data):
            if i % 3 == 0:  # sort the vertex coordinates
                start = ind * 3
                end = start + 3
                ObjLoader.buffer.extend(vertices[start:end])
                #print(len(indices_data))
            elif i % 3 == 1:  # sort the texture coordinates
                start = ind * 2
                end = start + 2
                ObjLoader.buffer.extend(textures[start:end])
            elif i % 3 == 2:  # sort the normal vectors
                start = ind * 3
                end = start + 3
                ObjLoader.buffer.extend(normals[start:end])

    @staticmethod
    def show_buffer_data(buffer):
        for i in range(len(buffer) // 8):
            start = i * 8
            end = start + 8
            print(buffer[start:end])

    @staticmethod
    def load_model(filename, sorted=True):
        vertices = []  # will contain all the vertex coordinates
        textures = []  # will contain all the texture coordinates
        normals = []   # will contain all the vertex normals

        index_data = []  # will contain all the vertex, texture, and normal indices
        indices = []     # will contain the indices for indexed drawing

        with open(filename, 'r') as file:
            line = file.readline()
            while line:
                values = line.split()
                if len(values) == 0:
                    # Skip empty lines or lines with just spaces
                    line = file.readline()
                    continue

                if values[0] == 'v':
                    ObjLoader.search_data(values[1:], vertices, 'v', 'float')
                elif values[0] == 'vt':
                    ObjLoader.search_data(values[1:], textures, 'vt', 'float')
                elif values[0] == 'vn':
                    ObjLoader.search_data(values[1:], normals, 'vn', 'float')
                elif values[0] == 'f':
                    for value in values[1:]:
                        val = value.split('/')
                        if len(val) == 1:  # Only vertex index
                            ObjLoader.search_data([val[0]], index_data, 'f', 'int')
                        elif len(val) == 2:  # Vertex and texture index, no normal
                            ObjLoader.search_data([val[0]], index_data, 'f', 'int')
                            ObjLoader.search_data([val[1]], index_data, 'f', 'int')
                        elif len(val) == 3:  # Vertex, texture, and normal
                            ObjLoader.search_data([val[0]], index_data, 'f', 'int')
                            ObjLoader.search_data([val[1]], index_data, 'f', 'int')
                            ObjLoader.search_data([val[2]], index_data, 'f', 'int')
                        indices.append(int(val[0]))

                line = file.readline()

        if sorted:
            # use with glDrawArrays
            ObjLoader.create_sorted_vertex_buffer(index_data, vertices, textures, normals)

        buffer_copy = ObjLoader.buffer.copy()  # create a local copy of the buffer list
        ObjLoader.buffer = []  # reset the static field buffer

        # Automatically set texture path
        texture_path = ObjLoader.get_texture_path(filename)

        return np.array(indices, dtype='uint32'), np.array(buffer_copy, dtype='float32'), texture_path

    
    
    @staticmethod
    def get_texture_path(filename):
        # Get the base name of the file (without extension)
        base_name = os.path.splitext(filename)[0]
        # Assume the texture is a .jpg file with the same name
        texture_path = base_name + ".jpg"
        if os.path.exists(texture_path):
            return texture_path
        else:
            print(f"Warning: Texture file {texture_path} not found.")
            return None