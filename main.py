import sys
import pickle
import trimesh
import numpy as np
from scipy.sparse import coo_matrix, block_diag, csc_matrix
from scipy.sparse.linalg import lsqr


def load_obj(file_name):
    assert file_name.endswith('.obj')
    vertices = []
    with open(file_name) as file:
        line_ls = file.readlines()
        for line in line_ls:
            words = line.split()
            if words[0] == 'v':
                ver = words[1: 4]
                ver = [float(v) for v in ver]
                vertices.append(ver)
    return vertices


def get_connectivity(mesh):
    """
    vpv is a sparse matrix (#verts x #verts) where each nonzero element indicates a neighborhood relation. 
    i.e. if there is a nonzero element in position (15,12), vertex 15 will be connected by an edge to vertex 12.
    """
    mesh_f = mesh.faces
    mesh_v = mesh.vertices
    vpv = csc_matrix((len(mesh_v), len(mesh_v)))
    for i in range(3):
        IS = np.array(mesh_f[:, i])
        JS = np.array(mesh_f[:, (i+1) % 3])
        data = np.ones(len(IS))   
        ij = np.vstack((IS.flatten(), JS.flatten()))
        mtx = csc_matrix((data, ij), shape = vpv.shape)
        vpv = vpv + mtx + mtx.T
    return vpv


if __name__ == '__main__':

    # -----------  Implement Laplacian Surface Editting method from sketch ----------- #    
    vertex_ls = np.array(load_obj('blender/mid.obj'))

    with open('blender/anchor.pkl', 'rb') as file:
        anchors = pickle.load(file)
    anchor_id_ls = np.array(anchors)
    anchor_ls = vertex_ls[anchor_id_ls]      


    print('Solving matrix (mean weight)...')
    gar_mesh = trimesh.load('blender/in.obj')

    N = gar_mesh.vertices.shape[0]     # All points
    K = anchor_ls.shape[0]          # All anchors


    # ----------- Step 1: Build the linear system ------------- #
    vpv = get_connectivity(gar_mesh)
    data, row, col = [], [], []
    for i in range(N):
        vertex = gar_mesh.vertices[i]  
        neighbors = list(vpv[:, i].tocoo().row)   # Get the neighbors of one vertex

        degree = len(neighbors) 
        data += [degree] + [-1] * degree
        
        row += [i] * (degree + 1)   # append (degree + 1) items
        col += [i] + neighbors   # one vertex with its neighbors 


    # ------------- Step 2: Add Constraints to the Linear System ------------ #
    for i in range(K):
        data += [1]
        row += [N + i]
        col += [anchor_id_ls[i]]
    Ls = coo_matrix((data, (row, col)), shape=(N + K, N)).tocsr()  # Add the values in the same position


    # ------------- Step 3: Solve the Linear System ------------- #
    delta = Ls.dot(gar_mesh.vertices)    # Ls * orig_verts
    for i in range(K):
        delta[N + i] = anchor_ls[i]    # coordinates of anchors remains the same

    A = block_diag((Ls, Ls, Ls))
    print(delta.shape)
    b = np.hstack((delta[:, 0], delta[:, 1], delta[:, 2]))
    x = lsqr(A, b)[0].reshape((3, -1)).T

    gar_mesh.vertices = x
    gar_mesh.export('blender/out.obj')