import taichi as ti
import time
import math

# ti.core.toggle_advance_optimization(False)
# ti.init(debug=True)
# ti.init(arch=ti.gpu)
ti.init(arch=ti.cpu)

n_dim = 2
n_ptr = 8
dt    = 1e-3
ht    = dt * 0.5 # half of dt
max_num_particles = 2048
connection_radius = 0.9

bottom_y      = 0.05
top_y         = 0.95
left_x        = 0.05
right_x       = 0.95

gravity          = ti.Vector.field(n_dim, ti.f32, shape=())
num_particles    = ti.field(ti.i32, shape=())
spring_stiffness = ti.field(ti.f32, shape=())
particle_mass    = ti.field(ti.f32, shape=())
damping          = ti.field(ti.f32, shape=())
paused           = ti.field(ti.i32, shape=())


# rest_length[i, j] = 0 means i and j are not connected
rest_length  = ti.field(ti.f32, shape=(max_num_particles, n_ptr))
num_ptr      = ti.field(ti.i32, shape=())
num_adj      = ti.field(ti.i32, shape=max_num_particles)
adj_ptr      = ti.field(ti.i32, shape=(max_num_particles, n_ptr))
# rest_length  = ti.root.dense(ti.ij, max_num_particles)
position     = ti.Vector.field(n_dim, dtype=ti.f32, shape=max_num_particles) # position
velocity     = ti.Vector.field(n_dim, dtype=ti.f32, shape=max_num_particles) # velocity
new_velocity = ti.Vector.field(n_dim, dtype=ti.f32, shape=max_num_particles)
force        = ti.Vector.field(n_dim, dtype=ti.f32, shape=max_num_particles) # force 
new_force    = ti.Vector.field(n_dim, dtype=ti.f32, shape=max_num_particles) 
out_force    = ti.Vector.field(n_dim, dtype=ti.f32, shape=max_num_particles) 

# A @ new_velocity = b
A      = ti.Matrix.field(n_dim, n_dim, dtype=ti.f32, shape=(max_num_particles, n_ptr + 1))
b      = ti.Vector.field(n_dim, dtype=ti.f32, shape=max_num_particles)
r      = ti.Vector.field(n_dim, dtype=ti.f32, shape=max_num_particles)
new_dv = ti.Vector.field(n_dim, dtype=ti.f32, shape=max_num_particles)

resi_out = ti.field(ti.f32,shape=(2))

num_ptr[None]          = n_ptr
gravity[None]          = [0, 0] # -9.8]
num_particles[None]    = 0
particle_mass[None]    = 1
damping[None]          = 0
spring_stiffness[None] = 300
paused[None]           = False

@ti.func
def iterate():
    n = num_particles[None]
    for i in range(n):
        r[i] = b[i] * 1.0
        
    for i,m in ti.ndrange(n,num_ptr[None]) :
        j = adj_ptr[i,m]
        if j >= 0 :
            r[i] -= A[i, m] @ new_velocity[j]
                
    for i in range(n):
        ptr_i = num_ptr[None]
        # new_velocity[i] = A[i, i].inverse() @ r[i]
        new_dv[i] = A[i, ptr_i].inverse() @ r[i]
        # new_dv[i].x = (r[i].x - A[i,i][0,1] * new_velocity[i].y) / A[i, i][0,0]
        # new_dv[i].y = (r[i].y - A[i,i][1,0] * new_velocity[i].x) / A[i, i][1,1]
        
    for i in range(n):
        new_velocity[i] = new_dv[i] * 1.0


@ti.func
def resi() -> ti.f32:
    n = num_particles[None]
    res = 0.0
    
    for i in range(n):
        ptr_i = num_ptr[None]
        r[i] = b[i] * 1.0 - A[i,ptr_i] @ new_velocity[i]

    for i,m in ti.ndrange(n,num_ptr[None]) :
        j = adj_ptr[i,m]
        if j >= 0 :
            r[i] -= A[i, m] @ new_velocity[j]

    for i in range(n):
        res += r[i].x ** 2 
        res += r[i].y ** 2

    return res

@ti.kernel
def residual() -> ti.f32:
    return resi()

@ti.func
def solve_equation() :
    for no_loop in range(1) :
        for i in range(1000) :
            iterate()
            # print("----->----->",resi())
            resi_out[0] = resi()
            resi_out[1] = i
            # print("----->----->",resi_out[0])
            if resi_out[0] < 1e-5 :
                break

@ti.func
def f_ij(i,ptr_j) :
    j = adj_ptr[i,ptr_j]
    x_ij = position[j] - position[i] # from i to j
    x_d  = x_ij.normalized()
    x_n  = x_ij.norm()
    return spring_stiffness[None] * (x_n - rest_length[i, ptr_j]) * x_d

# (1 - l/xn) * (xj - xi)

# d_f/d_xi
# -[(1 - l/xn), 0] - x * l * [ x / xn^3, y / xn^3 ]
# -[0, (1 - l/xn)] - y * l * [ x / xn^3, y / xn^3 ]

# d_f/d_xj
# [(1 - l/xn), 0] + x * l * [ x / xn^3, y / xn^3 ]
# [0, (1 - l/xn)] + y * l * [ x / xn^3, y / xn^3 ]

# I * (1 - l/xn) + l/xn * x_o
# I + l/nx * (x_o - I)

# df / dx
@ti.func
def dfj_ij(i,ptr_j) :
    j = adj_ptr[i,ptr_j]
    x_ij = position[j] - position[i] # from i to j
    x_d  = x_ij.normalized()
    x_n  = x_ij.norm()
    x_o  = x_d.outer_product(x_d)
    x_e = ti.Matrix([[1,0],[0,1]])
    l = rest_length[i,ptr_j]
    res =  x_e + l / x_n * (x_o - x_e)
    return spring_stiffness[None] * res

@ti.func
def collide_box() :
    n = num_particles[None]
    # Collide with box
    for i in range(n):
        if position[i].y < bottom_y:
            position[i].y  = bottom_y
            # new_force[i].y = 0 # abs(new_force[i].y)
            if velocity[i].y < 0 :
                velocity[i].y  = 0 # abs(velocity[i].y)
            # velocity[i].y = 0
        if position[i].y > top_y:
            position[i].y  = top_y
            # new_force[i].y = 0 # -abs(new_force[i].y)
            if velocity[i].y > 0 :
                velocity[i].y  = 0 # -abs(velocity[i].y)
            # velocity[i].y = 0
        if position[i].x < left_x:
            position[i].x  = left_x
            # if new_force[i].x < 0 :
            #     new_force[i].x = 0 # abs(new_force[i].x)
            if velocity[i].x < 0 :
                velocity[i].x  = 0 # abs(velocity[i].x)
            # velocity[i].x = 0
        if position[i].x > right_x:
            position[i].x  = right_x
            # new_force[i].x = 0 # -abs(new_force[i].x)
            if velocity[i].x > 0 :
                velocity[i].x  = 0 # -abs(velocity[i].x)
            # velocity[i].x = 0
@ti.func
def substep_jacobi_semi():
    n = num_particles[None]
    n_p = num_ptr[None]
    ht2 = ht * ht

    # calc force
    for i in range(n) :
        new_force[i] = gravity * particle_mass[None] + out_force[i]
    for i,m in ti.ndrange(n,n_p) :
        j = adj_ptr[i,m]
        if j >= 0 :
            new_force[i] += f_ij(i,m) 

    # collide_box()

    # init new velocity
    for i in range(n) :
        new_velocity[i] = velocity[i] + new_force[i] / particle_mass[None] * dt
        force[i] = new_force[i]

    mass = ti.Matrix([[particle_mass[None],0.0],[0.0,particle_mass[None]]])

    # fill A b
    for i in range(n) :
        ptr_i = num_ptr[None]
        A[i,ptr_i] *= 0.0
    for i,m in ti.ndrange(n,n_p) :
        j = adj_ptr[i,m]
        if j >= 0 :
            A[i,m] *= 0.0

    for i in range(n) :
        ptr_i = num_ptr[None]
        A[i,ptr_i] += mass
    for i,m in ti.ndrange(n,n_p) :
        ptr_i = num_ptr[None]
        j = adj_ptr[i,m]
        if j >= 0 :
            A[i,ptr_i] += ht2 * dfj_ij(i,m)
            A[i,m] -= ht2 * dfj_ij(i,m)

    for i in range(n) :
        b[i] = new_force[i] * ht * 2 + mass @ velocity[i]
    for i,m in ti.ndrange(n,n_p) :
        j = adj_ptr[i,m]
        if j >= 0 :
            b[i] -= ht2 * dfj_ij(i,m) @ velocity[i] 
            b[i] += ht2 * dfj_ij(i,m) @ velocity[j] 
                
    # solve equation for new velocity
    solve_equation()

    # Compute new position
    for i in range(n) :
        if position[i].y > bottom_y :
            position[i] += (velocity[i] + new_velocity[i]) * ht
            velocity[i] = new_velocity[i]
        # velocity[i] *= ti.exp(-dt * damping[None]) # damping

@ti.kernel
def step_jacobi():
    for no_loop in range(1) :
        for step in range(80):
            substep_jacobi_semi()
            # substep_jacobi()
        
@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32): # Taichi doesn't support using Matrices as kernel arguments yet
    new_particle_id = num_particles[None]
    position[new_particle_id] = [pos_x, pos_y]
    velocity[new_particle_id] = [0, 0]
    num_particles[None] += 1

@ti.kernel
def hit_particle(pos_x: ti.f32, pos_y: ti.f32): 
    n = num_particles[None]
    dist_x = 1e9
    for no_loop in range(1) :
        for i in range(n) :
            dist = position[i].x - pos_x
            if abs(dist_x) > abs(dist) :
                dist_x = dist
    print('dist_x:',dist_x)
    for i in range(n) :
        dist = (position[i] - ti.Vector([pos_x,pos_y])).norm()
        # dist = abs(position[i].y - pos_y)
        if dist < 0.05 and position[i].y > bottom_y :
            out_force[i] += ti.Vector([dist_x,0.0]) * 30
            # position[i] += ti.Vector([dist_x,0.0])

@ti.kernel
def hit_clear(): 
    n = num_particles[None]
    for i in range(n) :
        out_force[i] *= 0


@ti.kernel
def conn_particle(pos_i: ti.i32, pos_j: ti.i32) :
    dist = (position[pos_i] - position[pos_j]).norm()
    ptr_j = num_adj[pos_i]
    rest_length[pos_i, ptr_j] = dist # 0.1
    adj_ptr[pos_i,ptr_j] = pos_j
    num_adj[pos_i] += 1

    ptr_i = num_adj[pos_j]
    rest_length[pos_j, ptr_i] = dist # 0.1
    adj_ptr[pos_j, ptr_i] = pos_i
    num_adj[pos_j] += 1

    

def init() :
    num_particles[None] = 0
    rest_length.fill(0)
    num_adj.fill(0)
    adj_ptr.fill(-1)
    n_xs = 2
    n_ys = 40
    for i in range(n_xs) :
        for j in range(n_ys) :
            new_particle(0.52 + i * 0.02, 0.00 + bottom_y + j * 0.02)
            if i > 0 :
                conn_particle(i*n_ys+j,(i-1)*n_ys+(j+0))
            if j > 0 :
                conn_particle(i*n_ys+j,(i-0)*n_ys+(j-1))
            if i > 0 and j > 0 :
                conn_particle(i*n_ys+j,(i-1)*n_ys+(j-1))
                conn_particle((i-0)*n_ys+(j-1),(i-1)*n_ys+(j-0))

init()

gui = ti.GUI('Mass Spring System', res=(512, 512), background_color=0xdddddd)

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]
        elif e.key == ti.GUI.LMB:
            # new_particle(e.pos[0], e.pos[1])
            hit_particle(e.pos[0], e.pos[1])
        elif e.key == 'c':
            init()
        elif e.key == 's':
            if gui.is_pressed('Shift'):
                spring_stiffness[None] /= 1.1
            else:
                spring_stiffness[None] *= 1.1
        elif e.key == 'd':
            if gui.is_pressed('Shift'):
                damping[None] /= 1.1
            else:
                damping[None] *= 1.1

                
    if not paused[None]:
        step_jacobi()
        hit_clear()
        print('resi out : ',resi_out[0],resi_out[1])
        # print(f'residual={residual():0.10f}')
    
    X = position.to_numpy()
    Y = rest_length.to_numpy()
    Z = adj_ptr.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=1)
    
    gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
    
    for i in range(num_particles[None]):
        #for j in range(i + 1, num_particles[None]):
        #    if rest_length[i, j] != 0:
        #        gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
        for ptr_j,j in enumerate(Z[i]) :
            if j >= 0 :
                dist = math.sqrt((X[i][0]-X[j][0])**2 + (X[i][1]-X[j][1])**2)
                ratio = (dist - Y[i][ptr_j])/Y[i][ptr_j]
                ratio *= 100
                ratio = max(-1.0, ratio)
                ratio = min( 1.0, ratio)
                if ratio < 0 :
                    color = int(-ratio * 255)
                    color *= 0x10000
                else :
                    color = int( ratio * 255)
                gui.line(begin=X[i], end=X[j], radius=1.5, color=color + 0x000000 )
    gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.99), color=0x0)
    gui.text(content=f'S: Spring stiffness {spring_stiffness[None]:.1f}', pos=(0.5, 0.99), color=0x0)
    # gui.text(content=f'D: damping {damping[None]:.2f}', pos=(0, 0.85), color=0x0)
    gui.show()


